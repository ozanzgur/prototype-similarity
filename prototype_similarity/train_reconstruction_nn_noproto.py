# %%
# Experiment: Train 1x1 convs instead of prototypes
# Initialize
# 0.9641
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

DATASET_CIFAR10 = "cifar10"
DATASET_CIFAR100 = "cifar100"
DATASET_TINYIMAGENET = "tin"
DATASET_IMAGENET = "imagenet"
DATASET_IMAGENET200 = "imagenet200"

ID_DATASET = DATASET_CIFAR10
OOD_TRAIN_DATASET = DATASET_TINYIMAGENET

normalization_dict = {
    'cifar10': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
    'cifar100': [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
    'imagenet': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    'imagenet200': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    'covid': [[0.4907, 0.4907, 0.4907], [0.2697, 0.2697, 0.2697]],
    'aircraft': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
    'cub': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
    'cars': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
}
mean, std = normalization_dict[ID_DATASET]

    
# %%

activations = {}
def get_activation(name):
    def hook(model, input, output):
        output = output.detach()
        if len(output.shape) == 2:
            output = output.unsqueeze(-1).unsqueeze(-1)
        #output[output < 0] = 0
        activations[name] = output
    return hook

# %%

dataset_path = "/home/ozan/projects/git/prototype-similarity/data"
if ID_DATASET == DATASET_CIFAR10:
    from openood.networks import ResNet18_32x32

    model_path = "/home/ozan/projects/git/prototype-similarity/model/cifar10_res18_v1.5/cifar10_resnet18_32x32_base_e100_lr0.1_default/s1/best.ckpt"

    net = ResNet18_32x32(num_classes=10)
    net.load_state_dict(torch.load(model_path))
    net.eval(); net.cuda();
    #model = ReconstructModel(512).cuda()
    
    #net.layer1[1].conv1.register_forward_hook(get_activation('b1_relu1'))
    #net.layer2[1].conv1.register_forward_hook(get_activation('b2_relu1'))
    #net.layer3[1].conv1.register_forward_hook(get_activation('b3_relu1'))
    net.layer4[1].conv1.register_forward_hook(get_activation('b4_relu1'))
    net.fc.register_forward_hook(get_activation('fc'))

    test_transform = tt.Compose([tt.ToTensor(), tt.Normalize(mean, std)])
    train_dataset = torchvision.datasets.CIFAR10(dataset_path, train=True, transform=test_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(dataset_path, train=False, transform=test_transform, download=True)
    batch_size = 32

elif ID_DATASET == DATASET_CIFAR100:
    from openood.networks import ResNet18_32x32

    model_path = "/home/ozan/projects/git/prototype-similarity/model/cifar100_res18_v1.5/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt"
    
    net = ResNet18_32x32(num_classes=100)
    net.load_state_dict(torch.load(model_path))
    net.eval(); net.cuda();
    model = ReconstructModel(512).cuda()
    net.layer4[0].conv1.register_forward_hook(get_activation('b4_relu1'))

    test_transform = tt.Compose([tt.ToTensor(), tt.Normalize(mean, std)])
    train_dataset = torchvision.datasets.CIFAR100(dataset_path, train=True, transform=test_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR100(dataset_path, train=False, transform=test_transform, download=True)
    batch_size = 32

elif ID_DATASET == DATASET_IMAGENET200:
    model_path = "/home/ozan/projects/git/prototype-similarity/model/imagenet200_res18_v1.5/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best.ckpt"
    
    net = ResNet18_224x224(num_classes=200)
    net.load_state_dict(torch.load(model_path))
    net.eval(); net.cuda();
    model = ReconstructModel(512).cuda()
    net.layer4[0].conv1.register_forward_hook(get_activation('b4_relu1'))

    test_transform = tt.Compose([tt.ToTensor(), tt.Normalize(mean, std)])
    train_dataset = torchvision.datasets.CIFAR100(dataset_path, train=True, transform=test_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR100(dataset_path, train=False, transform=test_transform, download=True)
    batch_size = 32


elif ID_DATASET == DATASET_IMAGENET:
    from datasets import load_dataset
    from openood.networks import ResNet50
    from torchvision.models import ResNet50_Weights
    from torch.hub import load_state_dict_from_url

    net = ResNet50()
    weights = ResNet50_Weights.IMAGENET1K_V1
    net.load_state_dict(load_state_dict_from_url(weights.url))
    test_transform = weights.transforms()
    net.eval(); net.cuda()
    net.layer4[0].conv1.register_forward_hook(get_activation('b4_relu1'))

    #train_dataset = torchvision.datasets.ImageNet(dataset_path, train=True, transform=test_transform, download=True)
    #test_dataset = torchvision.datasets.ImageNet(dataset_path, train=False, transform=test_transform, download=True)
    train_dataset = load_dataset("imagenet-1k", split='train')
    testdataset = load_dataset("imagenet-1k", split='test')
    train_dataset.set_transform(test_transform)
    testdataset.set_transform(test_transform)
    batch_size = 4

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
)

"""logger = TensorBoardLogger(save_dir="training_logs")
trainer = pl.Trainer(max_epochs=3, logger=logger, accumulate_grad_batches=1, precision=32, log_every_n_steps=5)
net.eval()
trainer.fit(model, train_loader)
model.eval(); model.cuda();"""

# %%
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from datasets import load_dataset

class ToRGB:
    def __call__(self, img):
        return img.convert('RGB')
    
def collate_fn(examples):
    images = []
    labels = []
    for example in examples:
        images.append((example["image"]))
        labels.append(example["label"])
        
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels

class RandomResizeCenterCrop:
    def __init__(self, size_options):
        self.size_options = size_options
    
    def __call__(self, img):
        # Select a random size from the options
        selected_size = random.choice(self.size_options)
        
        # Resize the image
        resize_transform = tt.Resize(selected_size)
        img_resized = resize_transform(img)
        return img_resized

class MetamodelDataset(torch.utils.data.Dataset):
    def __init__(self):
        dataset1 = load_dataset('Maysee/tiny-imagenet', split='train')
        test_idx = np.random.permutation(len(dataset1))
        test_idx = [int(i) for i in test_idx[:1000]]
        dataset1 = torch.utils.data.Subset(dataset1, indices=test_idx)
        self.dataset1 = dataset1

        self.dataset2 = torchvision.datasets.CIFAR10(dataset_path, 
            train=True, transform=None, download=True)
        self.length1 = len(self.dataset1)
        self.length2 = len(self.dataset2)
        self.total_length = self.length1 + self.length2

        transforms_cifar10 = [
            ToRGB(),
            #tt.RandomResizedCrop((32,32), scale=(0.9, 1.1)),
            tt.ToTensor(),
            tt.Normalize(mean, std)
        ]
        self.transforms_cifar10 = tt.Compose(transforms_cifar10)
        transforms_tin = [
            ToRGB(),
            tt.RandomResizedCrop((32,32), scale=(0.4, 1.3)),
            #RandomResizeCenterCrop([88, 76, 58, 52, 40, 28]),
            tt.CenterCrop(32),
            tt.ToTensor(),
            tt.Normalize(mean, std)
        ]
        self.transforms_tin = tt.Compose(transforms_tin)
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, index):
        if index < self.length1:
            # Fetch from the first dataset and assign label 1
            data = self.dataset1[index]["image"]
            label = torch.tensor([1, 0], dtype=torch.float32)
            data = self.transforms_tin(data)
        else:
            # Fetch from the second dataset and assign label 0
            data, target = self.dataset2[index - self.length1]
            label = torch.tensor([0, 1], dtype=torch.float32)
            data = self.transforms_cifar10(data)

        return data, label
    
# %%
class FeatureExtractor(nn.Module):
    def __init__(self, n_channels, n_features):
        super(FeatureExtractor, self).__init__()
        self.conv1x1 = nn.Conv2d(n_channels, n_features, kernel_size=1)

    def forward(self, x):
        # x is expected to be of shape (batch_size, n_channels, h, w)
        features = self.conv1x1(x)
        features = features.mean(dim=-1).mean(dim=-1)
        return features

class MLP(pl.LightningModule):
    def __init__(self, hidden_size=100, output_size=2):
        super(MLP, self).__init__()
        self.automatic_optimization = False
        #self.extractor1 = FeatureExtractor(64, 128)
        #self.extractor2 = FeatureExtractor(128, 256)
        #self.extractor3 = FeatureExtractor(256, 512)
        self.extractor4 = FeatureExtractor(512, 1024)
        self.extractor5 = FeatureExtractor(10, 40)

        self.fc1 = nn.Linear(1024+40, hidden_size) # 128+256+512+
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def get_intermediate_acts(self, net, x):
        with torch.no_grad():
            _ = net(x.cuda())
        acts = [activations[n] for n in ["b4_relu1", "fc"]] # "b1_relu1", "b2_relu1", "b3_relu1", 
        return acts

    def forward(self, x_img):
        acts = self.get_intermediate_acts(net, x_img)
        #f1 = self.extractor1(acts[0]) # batch_size, n_feat
        #f2 = self.extractor2(acts[1])
        #f3 = self.extractor3(acts[2])
        f4 = self.extractor4(acts[0])
        f5 = self.extractor5(acts[1])
        features = torch.cat([f4, f5], dim=1) # f1, f2, f3, 

        out = self.fc1(features)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
    
    def training_step(self, train_batch, batch_idx):
        opt_reg = self.optimizers()
        x_img, y = train_batch
        y_pred = self.forward(x_img)
        train_loss_mlp = self.loss(y_pred, y)
        self.log('train_loss_mlp', train_loss_mlp)

        opt_reg.zero_grad()
        self.manual_backward(train_loss_mlp)
        opt_reg.step()
    
    def configure_optimizers(self):
        opt_reg = torch.optim.Adam(self.parameters(), lr=1e-3) # , weight_decay=1e-1
        sch_reg = torch.optim.lr_scheduler.MultiStepLR(opt_reg, [10, 20], gamma=0.1)
        return [opt_reg], [sch_reg]
    
    def on_train_epoch_end(self) -> None:
        sch_reg = self.lr_schedulers()
        sch_reg.step()
        return super().on_train_epoch_end()


metamodel = MLP().cuda()
metamodel_dataset = MetamodelDataset()
metamodel_loader = torch.utils.data.DataLoader(
    metamodel_dataset,
    batch_size=32, 
    shuffle=True
)

# Train the model
logger = TensorBoardLogger(save_dir="training_logs_metamodel")
trainer = pl.Trainer(max_epochs=30, logger=logger, accumulate_grad_batches=1, precision=32, log_every_n_steps=5)
trainer.fit(metamodel, metamodel_loader)
metamodel.eval(); metamodel.cuda();

# %%

# Make predictions on the test set
with torch.no_grad():
    preds = []
    labels = []
    for x_batch, y_batch in tqdm(metamodel_loader):
        preds.append(metamodel(x_batch.cuda()).cpu())
        labels.append(y_batch)
    
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    labels = labels[:, 1]
    y_pred = preds[:, 1] > 0.5
    y_pred_proba = preds[:, 1]
    y_pred = y_pred.numpy()
    y_pred_proba = y_pred_proba.numpy()

# Calculate accuracy
accuracy = accuracy_score(labels, y_pred)
print("Accuracy:", accuracy)

auc = roc_auc_score(labels, y_pred_proba)
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
        y_pred = metamodel(data.cuda())
        pred = y_pred[:, 1] > 0.5
        conf = y_pred[:, 1]
        return pred, conf

# %%
from openood.evaluation_api import Evaluator

transform = tt.Compose([
    ToRGB(),
    tt.Resize((32, 32)),
    tt.CenterCrop(32),
    tt.ToTensor(),
    tt.Normalize(mean, std)
])

# Initialize an evaluator and evaluate
evaluator = Evaluator(net, id_name=ID_DATASET,
    preprocessor=transform, postprocessor=PrototypePostprocessor(None), batch_size=50)
metrics = evaluator.eval_ood()
# %%
