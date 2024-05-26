# %%
# Initialize
import torch
import torchvision
import numpy as np
import torchvision.transforms as tt
import random
from pathlib import Path
import sys
import os
import pickle
import torch.nn.functional as F
from tqdm import tqdm
import pytorch_lightning as pl
from torch import nn
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger

from prototype_matching_model import PrototypeMatchingModel

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)

N_PROTOTYPES = 250

MODEL_CIFAR10 = 0
MODEL_CIFAR100 = 1

MODEL_TYPE = MODEL_CIFAR100

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

test_transform = tt.Compose([tt.ToTensor(), tt.Normalize(mean, std)])

class ReconstructModel(pl.LightningModule):
    def __init__(self, input_dim=128):
        super(ReconstructModel, self).__init__()
        self.automatic_optimization = False
        self.loss = nn.MSELoss()

        self.prototype_matcher = PrototypeMatchingModel(input_dim=input_dim, num_prototypes=N_PROTOTYPES)

    def forward(self, x):
        reconstructed_x, indices = self.prototype_matcher(x)
        output = reconstructed_x# + self.output_prototype # 
        return output, indices

    def training_step(self, train_batch, batch_idx):
        opt_reg = self.optimizers()
        x, _ = train_batch
        with torch.no_grad():
            _ = net(x.cuda())

        x_rec = activations["b4_relu1"]
        y_rec = activations["b4_relu1"]
        pred_rec, _ = self.forward(x_rec)

        train_loss = self.loss(pred_rec, y_rec)# + self.loss2(pred_rec, y_rec)
        self.log('train_loss', train_loss)

        opt_reg.zero_grad()
        self.manual_backward(train_loss)
        opt_reg.step()

    def configure_optimizers(self):
        opt_reg = torch.optim.Adam(self.parameters(), lr=3e-2) # , weight_decay=1e-1
        sch_reg = torch.optim.lr_scheduler.MultiStepLR(opt_reg, [1, 3, 5], gamma=0.1)
        return [opt_reg], [sch_reg] # sch_reg, sch_cls

    def on_train_epoch_end(self) -> None:
        sch_reg = self.lr_schedulers()
        sch_reg.step()
        return super().on_train_epoch_end()
    
# %%

activations = {}
def get_activation(name):
    def hook(model, input, output):
        output = output.detach()
        output[output < 0] = 0
        activations[name] = output
    return hook

def reconstruct_score(model, x):
    with torch.no_grad():
        _ = net(x.cuda())
    act = activations["b4_relu1"]
    h, w = act.shape[-2:]
    batch_tokens = act.flatten(start_dim=2).unsqueeze(-1) # batch_size, n_features, h*w, 1
    #print(batch_tokens.shape)

    #emb_size, n_prot = model.prototype_matcher.prototype_bank.shape
    prot = model.prototype_matcher.prototype_bank.T.unsqueeze(0).unsqueeze(2) # 1, emb_size, 1, n_prot
    similarities_cos = (F.normalize(batch_tokens, dim=1) * F.normalize(prot, dim=1)).sum(dim=1) # batch_size, h*w, n_prot
    similarities = torch.square(batch_tokens - prot).mean(dim=1)#.mean(dim=-2)
    #similarities = torch.mean(torch.mean(similarities, dim=1), dim=1) # batch_size

    similarities = similarities.mean(dim=1, keepdim=True)
    similarities_cos = similarities_cos.mean(dim=1, keepdim=True) # TODO: Try mean
    return similarities, similarities_cos

# %%
from openood.networks import ResNet18_32x32

# Load an ImageNet-pretrained model from torchvision
#model_path = "/home/ozan/projects/git/prototype-similarity/model/cifar10_res18_v1.5/cifar10_resnet18_32x32_base_e100_lr0.1_default/s1/best.ckpt"
model_path = "/home/ozan/projects/git/prototype-similarity/model/cifar100_res18_v1.5/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt"

dataset_path = "/home/ozan/projects/git/prototype-similarity/data"

net = ResNet18_32x32(num_classes=100)
net.load_state_dict(torch.load(model_path))
net.eval(); net.cuda();
model = ReconstructModel(512).cuda()

net.layer4[0].conv1.register_forward_hook(get_activation('b4_relu1'))

if MODEL_TYPE == MODEL_CIFAR10:
    train_dataset = torchvision.datasets.CIFAR10(dataset_path, train=True, transform=test_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(dataset_path, train=False, transform=test_transform, download=True)
else:
    train_dataset = torchvision.datasets.CIFAR100(dataset_path, train=True, transform=test_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR100(dataset_path, train=False, transform=test_transform, download=True)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True
)

logger = TensorBoardLogger(save_dir="training_logs")
trainer = pl.Trainer(max_epochs=2, logger=logger, accumulate_grad_batches=1, precision=32, log_every_n_steps=5)
net.eval()
trainer.fit(model, train_loader)
model.eval(); model.cuda();

# %%

def score_dataset(net, loader):
    scores = []
    scores_cos = []
    print("Scoring dataset")
    with torch.no_grad():
        for x_batch, _ in tqdm(loader):
            s, s_cos = reconstruct_score(model, x_batch.cuda())
            scores.append(s.cpu().numpy())
            scores_cos.append(s_cos.cpu().numpy())
    scores = np.concatenate(scores, axis=0).astype(np.float16)
    scores_cos = np.concatenate(scores_cos, axis=0).astype(np.float16)
    return scores, scores_cos

test_id_scores, test_id_scores_cos = score_dataset(net, test_loader)
train_id_scores, train_id_scores_cos = score_dataset(net, train_loader)

# %%
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


def get_ood_training_dataset():
    from datasets import load_dataset

    class ToRGB:
        def __call__(self, img):
            return img.convert('RGB')

    transform = tt.Compose([
        ToRGB(),
        #tt.Resize((32, 32)),
        tt.CenterCrop(32),
        tt.ToTensor(),
        tt.Normalize(mean, std)
    ])
    def transform_fn(examples):
        examples["image"] = [transform(image) for image in examples["image"]]
        return examples

    def collate_fn(examples):
        images = []
        labels = []
        for example in examples:
            images.append((example["image"]))
            labels.append(example["label"])
            
        images = torch.stack(images)
        labels = torch.tensor(labels)
        return images, labels

    test_dataset = load_dataset('Maysee/tiny-imagenet', split='train')
    #test_idx = np.random.permutation(len(test_dataset))
    #test_idx = [int(i) for i in test_idx]
    test_dataset.set_transform(transform_fn)
    #test_dataset = Subset(test_dataset, indices=test_idx)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32, 
        shuffle=False,
        collate_fn=collate_fn
    )
    return test_loader

# Get the ood training data
print("Getting the ood training dataset for id classification")
ood_training_loader = get_ood_training_dataset()
ood_training_scores, ood_training_scores_cos  = score_dataset(model, ood_training_loader)
ood_mean_scores = ood_training_scores.mean(axis=1)
ood_mean_scores_cos = ood_training_scores_cos.mean(axis=1)
X_ood = np.concatenate([ood_mean_scores, ood_mean_scores_cos], axis=1)
y_ood = np.zeros(len(X_ood), dtype=np.int32)

# Get the id training data
id_mean_scores = train_id_scores.mean(axis=1)
id_mean_scores_cos = train_id_scores_cos.mean(axis=1)
X_id = np.concatenate([id_mean_scores, id_mean_scores_cos], axis=1)
y_id = np.ones(len(X_id), dtype=np.int32)

X = np.concatenate([X_ood, X_id], axis=0)
y = np.concatenate([y_ood, y_id], axis=0)
print(f"X: {X.shape}, y: {y.shape}")

# Split data into train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X
y_train = y
X_test = X
y_test = y

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# %%
metamodel = GradientBoostingClassifier(
    random_state=42,
    n_estimators=500,
    learning_rate=3e-2,
    n_iter_no_change=5,
    subsample=0.75,
    max_features='sqrt',
    verbose=True
    )

# Train the model
metamodel.fit(X_train, y_train)

# Make predictions on the test set
y_pred = metamodel.predict(X_test)
y_pred_proba = metamodel.predict_proba(X_test)[:, 1]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

auc = roc_auc_score(y_test, y_pred_proba)
print("AUC:", auc)

# %%

from openood.postprocessors import BasePostprocessor
import openood.utils.comm as comm

class PrototypePostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(PrototypePostprocessor, self).__init__(config)
        self.APS_mode = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net, data):
        s, s_cos = reconstruct_score(model, data.cuda())
        s = s[:, 0].cpu().numpy(); s_cos = s_cos[:, 0].cpu().numpy()
        features = np.concatenate([s, s_cos], axis=1)
        features = scaler.transform(features)
        pred = metamodel.predict(features)
        conf = metamodel.predict_proba(features)[:, 1]
        return torch.tensor(pred), torch.tensor(conf)
    
    """def inference(self,
                  net,
                  data_loader,
                  progress: bool = True):
        pred_list, conf_list, label_list = [], [], []
        for batch in tqdm(data_loader,
                          disable=not progress or not comm.is_main_process()):
            data = batch['data'].cuda()
            label = batch['label'].cuda()
            pred, conf = self.postprocess(net, data)

            pred_list.append(pred)
            conf_list.append(conf)
            label_list.append(label)

        # convert values into numpy array
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        return pred_list, conf_list, label_list"""
    

# %%
from openood.evaluation_api import Evaluator

transform = tt.Compose([
    tt.CenterCrop(32),
    tt.ToTensor(),
    tt.Normalize(mean, std)
])

# Initialize an evaluator and evaluate
evaluator = Evaluator(net, id_name='cifar10',
    preprocessor=transform, postprocessor=PrototypePostprocessor(None))
metrics = evaluator.eval_ood()
# %%
