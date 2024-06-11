import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import pytorch_lightning as pl
from torch import nn
import numpy as np
from typing import List
import math

from prototype_matching_model import PrototypeMatchingModel

class FeatureExtractor(nn.Module):
    def __init__(self, n_channels:int, n_features:int, aggregate:bool=False):
        super(FeatureExtractor, self).__init__()
        self.aggregate=aggregate
        self.conv1x1 = nn.Conv2d(n_channels, n_features, kernel_size=1)

    def forward(self, x):
        # x is expected to be of shape (batch_size, n_channels, h, w)
        features = self.conv1x1(x)
        if self.aggregate:
            features = features.max(dim=-1).values.max(dim=-1).values
        else:
            features = features.flatten(start_dim=1)
        return features

class ReconstructModel(pl.LightningModule):
    def __init__(self, 
            backbone, 
            input_dims:List[int], 
            num_proto:List[int], 
            act_names:List[str], 
            spatial_avg_features:bool, 
            output_act_size:int=4,
            is_conv_method:bool=False
        ):
        super(ReconstructModel, self).__init__()
        #self.automatic_optimization = False
        self.loss = nn.MSELoss()
        self.is_conv_method = is_conv_method

        if not is_conv_method:
            self.prototype_matchers = nn.ModuleList([
                PrototypeMatchingModel(input_dim=dims, num_prototypes=n_proto) for dims, n_proto in zip(input_dims, num_proto)
            ])
        else:
            self.prototype_matchers = nn.ModuleList([
                FeatureExtractor(n_channels=dims, n_features=n_proto, aggregate=spatial_avg_features) for dims, n_proto in zip(input_dims, num_proto)
            ])  

        self.act_names = act_names
        self.backbone = backbone
        self.output_act_size = output_act_size
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.spatial_avg_features = spatial_avg_features
        self.activations = {}

    def get_activation(self, name):
        def hook(model, input, output):
            output = output.detach()
            if len(output.shape) == 2:
                output = output.unsqueeze(-1).unsqueeze(-1)

            if output.shape[-2] > self.output_act_size:
                kernel_size = math.floor(output.shape[-2] / self.output_act_size)
                if kernel_size > 1:
                    output = F.avg_pool2d(output, kernel_size)

            self.activations[name] = output
        return hook
    
    def prototype_scores(self, x):
        if self.is_conv_method:
            return self.prototype_scores_conv_method(x)
        else:
            return self.prototype_scores_prot_method(x)
    
    def prototype_scores_conv_method(self, x):
        with torch.no_grad():
            _ = self.backbone(x.cuda())
        acts = [self.activations[n] for n in self.act_names]
        layer_scores = []

        for act, scorer in zip(acts, self.prototype_matchers):
            layer_scores.append(scorer(act))

        layer_scores = torch.cat(layer_scores, dim=-1)
        return layer_scores

    def prototype_scores_prot_method(self, x):
        with torch.no_grad():
            _ = self.backbone(x.cuda())
        acts = [self.activations[n] for n in self.act_names]

        layer_scores = []
        prots = [matcher.prototype_bank.T.unsqueeze(0).unsqueeze(2) for matcher in self.prototype_matchers]
        
        for act, prot in zip(acts, prots):
            h, w = act.shape[-2:]
            batch_tokens = act.flatten(start_dim=2).unsqueeze(-1) # batch_size, n_features, h*w, 1
            similarities_cos = (F.normalize(batch_tokens, dim=1) * F.normalize(prot, dim=1)).sum(dim=1) # batch_size, h*w, n_prot
            similarities = torch.square(batch_tokens - prot).mean(dim=1)
            
            if self.spatial_avg_features:
                similarities = similarities.mean(dim=1)
                similarities_cos = similarities_cos.max(dim=1).values
            else:
                similarities = similarities.flatten(start_dim=-2)#.mean(dim=1)
                similarities_cos = similarities_cos.flatten(start_dim=-2)#.max(dim=1).values
            layer_scores.append(similarities)
            layer_scores.append(similarities_cos)

        layer_scores = torch.cat(layer_scores, dim=-1)
        return layer_scores
    
    def prototype_scores_loader(self, loader):
        scores = []
        print("Scoring dataset")
        with torch.no_grad():
            for x_batch, _ in tqdm(loader):
                scores.append(self.prototype_scores(x_batch.cuda()))

        scores = np.concatenate(scores, axis=0).astype(np.float16)
        return scores
    
    def forward(self, x):
        # Used for evaluating ID accuracy
        return self.backbone(x)

    def reconstruct(self, x):
        assert len(x) == len(self.prototype_matchers), f"{len(x)}, {len(self.prototype_matchers)}"
        reconstructed_output = [matcher(x_layer)[0] for matcher, x_layer in zip(self.prototype_matchers, x)]
        return reconstructed_output

    def training_step(self, train_batch, batch_idx):
        #opt_reg = self.optimizers()
        x, _ = train_batch
        with torch.no_grad():
            _ = self.backbone(x.cuda())

        orig_acts = [self.activations[n] for n in self.act_names]
        rec_acts = self.reconstruct(orig_acts)

        layer_losses = [self.loss(rec, orig) for rec, orig in zip(rec_acts, orig_acts)]
        train_loss = torch.sum(torch.stack(layer_losses))
        self.log('train_loss', train_loss)

        return train_loss

    def configure_optimizers(self):
        proto_params = self.prototype_matchers.parameters()
        opt_reg = torch.optim.Adam(proto_params, lr=1e-2) # , 0.1
        sch_reg = torch.optim.lr_scheduler.MultiStepLR(opt_reg, [3, 5], gamma=0.1)
        return [opt_reg], [sch_reg] # sch_reg, sch_cls

    def on_train_epoch_end(self) -> None:
        sch_reg = self.lr_schedulers()
        sch_reg.step()
        return super().on_train_epoch_end()