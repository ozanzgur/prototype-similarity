import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import pytorch_lightning as pl
from torch import nn
import numpy as np
from typing import List
import os
import math
from pathlib import Path

from prototype_matching_model import PrototypeMatchingModel

project_dir = str(Path(__file__).resolve().parents[1])

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

class ChannelMask(nn.Module):
    def __init__(self, n_channels:int):
        super(ChannelMask, self).__init__()
        self.n_channels = n_channels
        self.weights = nn.Parameter(torch.zeros((1, n_channels, 1, 1), dtype=torch.float32))

    def forward(self, x):
        return torch.sigmoid(self.weights) * x

class ReconstructModel(pl.LightningModule):
    def __init__(self, 
            backbone, 
            input_dims:List[int], 
            num_proto:List[int], 
            act_names:List[str], 
            spatial_avg_features:bool, 
            output_act_size:int=4,
            is_conv_method:bool=False,
            model_name:str=None
        ):
        super(ReconstructModel, self).__init__()
        #self.automatic_optimization = False
        self.loss = nn.MSELoss()
        self.model_name = model_name
        self.is_conv_method = is_conv_method

        if not is_conv_method:
            self.prototype_matchers = nn.ModuleList([
                PrototypeMatchingModel(input_dim=dims, num_prototypes=n_proto) for dims, n_proto in zip(input_dims, num_proto)
            ])
        else:
            self.prototype_matchers = nn.ModuleList([
                FeatureExtractor(n_channels=dims, n_features=n_proto, aggregate=spatial_avg_features) for dims, n_proto in zip(input_dims, num_proto)
            ])

        self.channel_masks_cos = nn.ModuleList([
                ChannelMask(n_channels=dims) for dims in input_dims
            ])
        self.channel_masks_l2 = nn.ModuleList([
                ChannelMask(n_channels=dims) for dims in input_dims
            ])

        self.act_names = act_names
        self.backbone = backbone
        self.output_act_size = output_act_size
        self.save_act_samples = False
        self.i_batch = 0 # Used for saving every nth batch activations
        self.save_period = 1000
        self.save_tag = "save"

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.spatial_avg_features = spatial_avg_features
        self.activations = {}

        self.feature_masking = False # Feature importance
        self.mask_layer_i = 0
        self.mask_prot_i = 0

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
        
    def reset_save_counter(self):
        self.i_batch = 0
    
    def prototype_scores_conv_method(self, x):
        with torch.no_grad():
            _ = self.backbone(x.cuda())
        acts = [self.activations[n] for n in self.act_names]
        layer_scores = []

        for act, scorer in zip(acts, self.prototype_matchers):
            layer_scores.append(scorer(act))

        layer_scores = torch.cat(layer_scores, dim=-1)
        return layer_scores
    
    def get_all_acts(self, x):
        with torch.no_grad():
            _ = self.backbone(x.cuda())
        acts = [self.activations[n] for n in self.act_names]
        return acts
    
    def save_acts(self, acts, tag=None):
        save_dir = f"{project_dir}/saved_acts"
        if self.model_name is not None:
            save_dir = save_dir + f"/{self.model_name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if tag is None:
            save_path = f'{save_dir}/saved_acts-{self.save_tag}-{self.i_batch}.pt'
        else:
            save_path = f'{save_dir}/saved_acts-{tag}-{self.save_tag}-{self.i_batch}.pt'
        torch.save(acts, save_path)
        print(f"Saving activations to: {save_path}")
    
    def prototype_scores_prot_method(self, x):
        acts = self.get_all_acts(x)

        if self.save_act_samples:
            #self.save_act_samples = False
            # Save a batch of activations
            if self.i_batch % self.save_period == 0:
                self.save_acts(acts, "prot_scores")
            self.i_batch += 1

        layer_scores = []
        prots = [matcher.prototype_bank.T.unsqueeze(0).unsqueeze(2) for matcher in self.prototype_matchers]
        masks_cos = self.channel_masks_cos
        masks_l2 = self.channel_masks_l2

        for i_layer, (act, prot, mask_cos, mask_l2) in enumerate(zip(acts, prots, masks_cos, masks_l2)):
            h, w = act.shape[-2:]
            batch_tokens = act.flatten(start_dim=2).unsqueeze(-1) # batch_size, n_features, h*w, 1
            similarities_cos = (F.normalize(batch_tokens, dim=1) * F.normalize(prot, dim=1)) # batch_size, n_ch, h*w, n_prot
            #similarities_cos = mask_cos(similarities_cos) # batch_size, n_ch, h*w, n_prot
            similarities_cos = similarities_cos.sum(dim=1) # batch_size, h*w, n_prot
            similarities = torch.square(batch_tokens - prot) # batch_size, n_ch, h*w, n_prot
            #similarities = mask_l2(similarities) # batch_size, n_ch, h*w, n_prot
            similarities = similarities.mean(dim=1) # batch_size, h*w, n_prot

            if self.spatial_avg_features:
                # feature_masking is not available for spatial_avg_features
                similarities_max = similarities.max(dim=1).values#.mean(dim=1)
                similarities_cos_max = similarities_cos.max(dim=1).values

                # similarities_min = similarities.min(dim=1).values#.mean(dim=1)
                # similarities_cos_min = similarities_cos.min(dim=1).values

                # similarities_mean = similarities.mean(dim=1)
                # similarities_cos_mean = similarities_cos.mean(dim=1)

                # print(similarities.shape)
                # print(similarities_cos.shape)
                # print(similarities_min.shape)
                # print(similarities_cos_min.shape)
                # print(similarities_mean.shape)
                # print(similarities_cos_mean.shape)

                layer_scores.append(similarities_max)
                layer_scores.append(similarities_cos_max)
                #layer_scores.append(similarities_min)
                #layer_scores.append(similarities_cos_min)
                #layer_scores.append(similarities_mean)
                #layer_scores.append(similarities_cos_mean)
            else:
                # Measure feature importance by masking each prototype's features one by one
                if self.feature_masking and self.mask_layer_i == i_layer:
                    similarities_cos[:, :, self.mask_prot_i] = 0
                    similarities[:, :, self.mask_prot_i] = 0

                similarities = similarities.flatten(start_dim=-2)#.mean(dim=1)
                similarities_cos = similarities_cos.flatten(start_dim=-2)#.max(dim=1).values # MODIFY to min + max

                layer_scores.append(similarities)
                layer_scores.append(similarities_cos)

        layer_scores = torch.cat(layer_scores, dim=-1)
        return layer_scores

    def disable_prototype_training(self):
        for param in self.prototype_matchers.parameters():
            param.requires_grad = False
    
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

        if self.save_act_samples:
            #self.save_act_samples = False
            # Save a batch of activations
            if self.i_batch % self.save_period == 0:
                self.save_acts(orig_acts, "train_orig_acts")
                self.save_acts(rec_acts, "train_rec_acts")
            self.i_batch += 1

        layer_losses = [self.loss(rec, orig) for rec, orig in zip(rec_acts, orig_acts)]
        train_loss = torch.sum(torch.stack(layer_losses))
        self.log('train_loss', train_loss)

        return train_loss

    def configure_optimizers(self):
        proto_params = self.prototype_matchers.parameters()
        opt_reg = torch.optim.Adam(proto_params, lr=1e-1) # , 0.1
        sch_reg = torch.optim.lr_scheduler.MultiStepLR(opt_reg, [3, 5], gamma=0.1)
        return [opt_reg], [sch_reg] # sch_reg, sch_cls

    def on_train_epoch_end(self) -> None:
        sch_reg = self.lr_schedulers()
        sch_reg.step()
        return super().on_train_epoch_end()