# %%

# More OOD data helped a lot in CIFAR100, add the graph to paper
# There is a tradeoff between far and near ood scores, why would that be?
# Initialize
# 0.9641

########### ID: cifar100 ##############################
#          FPR@95  AUROC  AUPR_IN  AUPR_OUT   ACC
# prot train id: cifar100
"""
cifar10     76.76  73.16    71.63     70.93 77.17
tin         46.84  85.09    89.69     77.27 77.17
nearood     61.80  79.13    80.66     74.10 77.17
mnist       32.63  90.17    73.03     98.06 77.17
svhn        15.73  96.40    93.13     98.51 77.17
texture     51.90  84.31    89.68     74.02 77.17
places365   59.26  80.81    60.35     92.53 77.17
farood      39.88  87.92    79.05     90.78 77.17
"""

# prot notrain, id: cifar100
"""
cifar10     80.43  70.02    67.71     68.71 77.17
tin         47.72  85.21    89.72     77.93 77.17
nearood     64.08  77.61    78.72     73.32 77.17
mnist       28.08  91.10    75.43     98.11 77.17
svhn        14.14  96.65    93.92     98.59 77.17
texture     54.04  85.41    90.30     77.76 77.17
places365   55.46  82.73    63.85     93.41 77.17
farood      37.93  88.97    80.88     91.97 77.17
"""

########### ID: cifar10 ##############################
# prot train, id: cifar10
"""
cifar100    30.13  91.72    92.38     90.26 94.63
tin         18.77  95.05    96.16     93.33 94.63
nearood     24.45  93.39    94.27     91.80 94.63
mnist        6.47  98.08    94.46     99.59 94.63
svhn         4.99  98.93    97.75     99.57 94.63
texture     13.40  96.79    97.83     95.00 94.63
places365   18.74  95.28    89.57     98.45 94.63
farood      10.90  97.27    94.90     98.15 94.63
"""

# prot notrain, id: cifar10
"""
cifar100    33.39  91.54    91.97     90.46 94.63
tin         17.86  95.43    96.41     94.03 94.63
nearood     25.62  93.48    94.19     92.25 94.63
mnist        8.07  97.79    92.15     99.57 94.63
svhn         3.62  99.26    98.40     99.72 94.63
texture     10.69  97.26    98.27     95.58 94.63
places365   17.07  96.07    90.53     98.77 94.63
farood       9.86  97.60    94.84     98.41 94.63
"""



# noprot 0.66 0.87
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

class ToRGB:
    def __call__(self, img):
        return img.convert('RGB')

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)

N_PROTOTYPES = 10

DATASET_CIFAR10 = "cifar10"
DATASET_CIFAR100 = "cifar100"
DATASET_TINYIMAGENET = "tin"
DATASET_IMAGENET = "imagenet"
DATASET_IMAGENET200 = "imagenet200"

ID_DATASET = DATASET_CIFAR10
OOD_TRAIN_DATASET = DATASET_CIFAR10

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

class ReconstructModel(pl.LightningModule):
    def __init__(self, input_dim=128):
        super(ReconstructModel, self).__init__()
        #self.automatic_optimization = False
        self.loss = nn.MSELoss()

        #self.prototype_matcher1 = PrototypeMatchingModel(input_dim=128, num_prototypes=100)
        self.prototype_matcher1_1 = PrototypeMatchingModel(input_dim=512, num_prototypes=N_PROTOTYPES)
        self.prototype_matcher1_2 = PrototypeMatchingModel(input_dim=512, num_prototypes=N_PROTOTYPES)
        self.prototype_matcher1_3 = PrototypeMatchingModel(input_dim=512, num_prototypes=N_PROTOTYPES)
        self.prototype_matcher1_4 = PrototypeMatchingModel(input_dim=512, num_prototypes=N_PROTOTYPES)
        self.prototype_matcher2 = PrototypeMatchingModel(input_dim=10, num_prototypes=40)

    def forward(self, x):
        reconstructed_x1, indices = self.prototype_matcher1_1(x[0])
        reconstructed_x2, indices = self.prototype_matcher1_2(x[1])
        reconstructed_x3, indices = self.prototype_matcher1_3(x[2])
        reconstructed_x4, indices = self.prototype_matcher1_4(x[3])
        reconstructed_x5, indices = self.prototype_matcher2(x[4])
        #reconstructed_x3, indices = self.prototype_matcher3(x[2])
        return reconstructed_x1, reconstructed_x2, reconstructed_x3, reconstructed_x4, reconstructed_x5

    def training_step(self, train_batch, batch_idx):
        #opt_reg = self.optimizers()
        x, _ = train_batch
        with torch.no_grad():
            _ = net(x.cuda())

        orig_acts = [activations[n] for n in ["b4_relu1", "b4_relu1", "b4_relu1", "b4_relu1", "fc"]]
        rec_acts = self.forward(orig_acts)

        layer_losses = [self.loss(rec, orig) for rec, orig in zip(rec_acts, orig_acts)]
        train_loss = torch.sum(torch.stack(layer_losses))
        self.log('train_loss', train_loss)

        #opt_reg.zero_grad()
        #self.manual_backward(train_loss)
        #opt_reg.step()
        return train_loss

    def configure_optimizers(self):
        opt_reg = torch.optim.Adam(self.parameters(), lr=1e-2) # , 0.1
        sch_reg = torch.optim.lr_scheduler.MultiStepLR(opt_reg, [3, 5], gamma=0.1)
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
        if len(output.shape) == 2:
            output = output.unsqueeze(-1).unsqueeze(-1)
        #output[output < 0] = 0
        activations[name] = output
    return hook

def reconstruct_score(model, x):
    with torch.no_grad():
        _ = net(x.cuda())
    acts = [activations[n] for n in ["b4_relu1", "b4_relu1", "b4_relu1", "b4_relu1", "fc"]]

    layer_scores = []
    prots = [
        model.prototype_matcher1_1.prototype_bank.T.unsqueeze(0).unsqueeze(2),
        model.prototype_matcher1_2.prototype_bank.T.unsqueeze(0).unsqueeze(2),
        model.prototype_matcher1_3.prototype_bank.T.unsqueeze(0).unsqueeze(2),
        model.prototype_matcher1_4.prototype_bank.T.unsqueeze(0).unsqueeze(2),
        model.prototype_matcher2.prototype_bank.T.unsqueeze(0).unsqueeze(2), # 1, emb_size, 1, n_prot
        #model.prototype_matcher3.prototype_bank.T.unsqueeze(0).unsqueeze(2)
    ]
    for act, prot in zip(acts, prots):
        h, w = act.shape[-2:]
        batch_tokens = act.flatten(start_dim=2).unsqueeze(-1) # batch_size, n_features, h*w, 1
        #print(batch_tokens.shape)

        #emb_size, n_prot = model.prototype_matcher.prototype_bank.shape
        similarities_cos = (F.normalize(batch_tokens, dim=1) * F.normalize(prot, dim=1)).sum(dim=1) # batch_size, h*w, n_prot
        similarities = torch.square(batch_tokens - prot).mean(dim=1)#.mean(dim=-2)
        #similarities = torch.mean(torch.mean(similarities, dim=1), dim=1) # batch_size

        similarities = similarities.flatten(start_dim=-2)#.mean(dim=1, keepdim=True)
        similarities_cos = similarities_cos.flatten(start_dim=-2)#.max(dim=1, keepdim=True).values
        layer_scores.append(similarities.cpu().numpy())
        layer_scores.append(similarities_cos.cpu().numpy())

    layer_scores = np.concatenate(layer_scores, axis=-1)
    return layer_scores

# %%

dataset_path = "/home/ozan/projects/git/prototype-similarity/data"
if ID_DATASET == DATASET_CIFAR10:
    from openood.networks import ResNet18_32x32

    model_path = "/home/ozan/projects/git/prototype-similarity/model/cifar10_res18_v1.5/cifar10_resnet18_32x32_base_e100_lr0.1_default/s1/best.ckpt"

    net = ResNet18_32x32(num_classes=10)
    net.load_state_dict(torch.load(model_path))
    net.eval(); net.cuda();
    model = ReconstructModel(512).cuda()
    
    net.layer1[1].register_forward_hook(get_activation('conv1'))
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

    net.layer1[1].register_forward_hook(get_activation('conv1'))
    net.layer4[1].conv1.register_forward_hook(get_activation('b4_relu1'))
    net.fc.register_forward_hook(get_activation('fc'))

    test_transform = tt.Compose([tt.ToTensor(), tt.Normalize(mean, std)])
    train_dataset = torchvision.datasets.CIFAR100(dataset_path, train=True, transform=test_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR100(dataset_path, train=False, transform=test_transform, download=True)
    batch_size = 16

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
"""test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
)"""
# %%
logger = TensorBoardLogger(save_dir="training_logs")
trainer = pl.Trainer(max_epochs=3, logger=logger, accumulate_grad_batches=1, precision=32, log_every_n_steps=5)
net.eval()
trainer.fit(model, train_loader)
model.eval(); model.cuda();

# %%

def score_dataset(net, loader):
    scores = []
    print("Scoring dataset")
    with torch.no_grad():
        for x_batch, _ in tqdm(loader):
            scores.append(reconstruct_score(model, x_batch.cuda()))

    scores = np.concatenate(scores, axis=0).astype(np.float16)
    return scores

train_id_scores = score_dataset(net, train_loader)

# %%
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from PIL import Image
import os.path as osp

class ImageDatasetFromFile(torch.utils.data.Dataset):
    def __init__(self, txt_file, img_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the text file with image paths and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_labels = []
        self.img_dir = img_dir
        with open(txt_file, 'r') as file:
            for line in file:
                # Split the line into filename and label
                parts = line.strip().split()
                if len(parts) == 2:
                    filename, label = parts
                    #if "/tin/test" in filename:
                    self.image_labels.append((filename, int(label)))
                else:
                    raise ValueError(f"Line in text file is not in expected format: {line}")

        self.transform = transform

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name, label = self.image_labels[idx]
        img_name = osp.join(self.img_dir, img_name)
        image = Image.open(img_name)  # Convert image to RGB

        if self.transform:
            image = self.transform(image)

        return image, label
    
def get_ood_training_dataset(resize=None):
    transforms = [
        ToRGB(),
        #tt.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        #tt.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        tt.CenterCrop(32),
        tt.ToTensor(),
        tt.Normalize(mean, std)
    ]
    if resize is not None:
        transforms.insert(1, tt.Resize((resize, resize)))
        
    transform = tt.Compose(transforms)
    ood_dataset = ImageDatasetFromFile(
        transform=transform,
        txt_file="/home/ozan/projects/git/prototype-similarity/data/benchmark_imglist/cifar10/train_tin597.txt",
        img_dir="/home/ozan/projects/git/prototype-similarity/data/images_classic"
    )
    ood_loader = torch.utils.data.DataLoader(
        ood_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return ood_loader

if False:#OOD_TRAIN_DATASET == DATASET_TINYIMAGENET:
    def get_ood_training_dataset(resize=None):
        from datasets import load_dataset

        transforms = [
            ToRGB(),
            #tt.Resize(32),
            tt.CenterCrop(32),
            tt.ToTensor(),
            tt.Normalize(mean, std)
        ]
        if resize is not None:
            #transforms.insert(1, tt.RandomPerspective(distortion_scale=0.5, p=0.5))
            #transforms.insert(1, tt.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10))
            transforms.insert(1, tt.Resize((resize, resize)))
            #transforms.insert(1, tt.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
            
        transform = tt.Compose(transforms)
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
        test_idx = np.random.permutation(len(test_dataset))
        test_idx = [int(i) for i in test_idx] # [:10000]
        test_dataset.set_transform(transform_fn)
        test_dataset = torch.utils.data.Subset(test_dataset, indices=test_idx)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=16, 
            shuffle=False,
            collate_fn=collate_fn
        )
        return test_loader
    
elif False: #OOD_TRAIN_DATASET == DATASET_CIFAR100:
    def get_ood_training_dataset(resize=None):
        transforms = [
            ToRGB(),
            tt.ToTensor(),
            tt.CenterCrop(32),
            tt.Normalize(mean, std)
        ]
        if resize is not None:
            transforms.insert(1, tt.Resize((resize, resize)))
            
        transform = tt.Compose(transforms)
        ood_dataset = torchvision.datasets.CIFAR100(dataset_path, train=True, transform=transform, download=True)
        ood_loader = torch.utils.data.DataLoader(
            ood_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        return ood_loader
    
elif False: #OOD_TRAIN_DATASET == DATASET_CIFAR10:
    def get_ood_training_dataset(resize=None):
        transforms = [
            ToRGB(),
            tt.ToTensor(),
            tt.CenterCrop(32),
            tt.Normalize(mean, std)
        ]
        if resize is not None:
            transforms.insert(1, tt.Resize((resize, resize)))
            
        transform = tt.Compose(transforms)
        ood_dataset = torchvision.datasets.CIFAR10(dataset_path, train=True, transform=transform, download=True)
        ood_loader = torch.utils.data.DataLoader(
            ood_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        return ood_loader
    
elif False:#OOD_TRAIN_DATASET == "":
    def get_ood_training_dataset(resize=None):
        transforms = [
            tt.Resize((384, 512)),
            test_transform
        ]
        if resize is not None:
            transforms.insert(1, tt.Resize((resize, resize)))

            transform = tt.Compose(transform)
        train_dataset = torchvision.datasets.CIFAR100(dataset_path, train=True, transform=transform, download=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        return train_loader

# Get the ood training data
print("Getting the ood training dataset for id classification")
ood_training_loader = get_ood_training_dataset()
ood_training_scores  = score_dataset(model, ood_training_loader)

#if OOD_TRAIN_DATASET == DATASET_TINYIMAGENET:
for resize in [88, 82, 76, 68, 58, 52, 47, 40, 34, 28]:
    print(f"Resize: {resize}")
    ood_training_loader = get_ood_training_dataset(resize=resize)
    ood_training_scores_aug  = score_dataset(model, ood_training_loader)
    ood_training_scores = np.concatenate([ood_training_scores, ood_training_scores_aug], axis=0)

# %%
ood_mean_scores = ood_training_scores#.mean(axis=1)

X_ood = np.concatenate([
    ood_mean_scores, 
], axis=1)
y_ood = np.zeros(len(X_ood), dtype=np.int32)

# Get the id training data
id_mean_scores = train_id_scores#.mean(axis=1)
X_id = np.concatenate([
    id_mean_scores
], axis=1)
y_id = np.ones(len(X_id), dtype=np.int32)

X = np.concatenate([X_ood, X_id], axis=0)
y = np.concatenate([y_ood, y_id], axis=0)
y_arr = np.zeros((len(y), 2))
y_arr[y == 0, 0] = 1
y_arr[y == 1, 1] = 1
print(f"X: {X.shape}, y: {y.shape}")

# Split data into train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X
y_train = y_arr
X_test = X
y_test = y_arr

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# %%
class MetamodelDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class MLP(pl.LightningModule):
    def __init__(self, input_size, hidden_size=50, output_size=2):
        super(MLP, self).__init__()
        self.automatic_optimization = False
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
    
    def training_step(self, train_batch, batch_idx):
        opt_reg = self.optimizers()
        x, y = train_batch
        
        y_pred = self.forward(x)
        train_loss_mlp = self.loss(y_pred, y)
        self.log('train_loss_mlp', train_loss_mlp)

        opt_reg.zero_grad()
        self.manual_backward(train_loss_mlp)
        opt_reg.step()
    
    def configure_optimizers(self):
        opt_reg = torch.optim.Adam(self.parameters(), lr=1e-2) # , weight_decay=1e-1
        sch_reg = torch.optim.lr_scheduler.MultiStepLR(opt_reg, [10, 20], gamma=0.1)
        return [opt_reg], [sch_reg]
    
    def on_train_epoch_end(self) -> None:
        sch_reg = self.lr_schedulers()
        sch_reg.step()
        return super().on_train_epoch_end()

# Train the OOD classifier
metamodel = MLP(input_size=X_train.shape[1]).cuda()
metamodel_dataset = MetamodelDataset(X_train, y_arr)
metamodel_loader = torch.utils.data.DataLoader(
    metamodel_dataset,
    batch_size=1000,
    shuffle=True
)

# Train the model
logger = TensorBoardLogger(save_dir="training_logs_metamodel")
trainer = pl.Trainer(max_epochs=2, logger=logger, accumulate_grad_batches=1, precision=32, log_every_n_steps=5)
trainer.fit(metamodel, metamodel_loader)
metamodel.eval(); metamodel.cuda();

# %%

# Make predictions on the test set
with torch.no_grad():
    metamodel_loader_noshuffle = torch.utils.data.DataLoader(
        metamodel_dataset,
        batch_size=1000, 
        shuffle=False
    )
    preds = [] 
    for x_batch, _ in tqdm(metamodel_loader_noshuffle):
        preds.append(metamodel(x_batch.cuda()).cpu())
    
    preds = torch.cat(preds)
    y_pred = preds[:, 1] > 0.5
    y_pred_proba = preds[:, 1]
    y_pred = y_pred.numpy()
    y_pred_proba = y_pred_proba.numpy()

# Calculate accuracy
accuracy = accuracy_score(y_test[:, 1], y_pred)
print("Accuracy:", accuracy)

auc = roc_auc_score(y_test[:, 1], y_pred_proba)
print("AUC:", auc)

# %%

from openood.postprocessors import BasePostprocessor
import openood.utils.comm as comm

class PrototypePostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(PrototypePostprocessor, self).__init__(config)
        self.APS_mode = False # hparam search

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net, data):
        s = reconstruct_score(model, data.cuda())#[:, 0]
        features = np.array(np.concatenate([
            s,
        ], axis=1))
        features = scaler.transform(features)
        #features[features > 0] = 0
        features = torch.tensor(features).cuda()
        y_pred = metamodel(features).cpu()
        pred = y_pred[:, 1] > 0.5
        conf = y_pred[:, 1]
        return pred, conf

# %%
from openood.evaluation_api import Evaluator

transform = tt.Compose([
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
