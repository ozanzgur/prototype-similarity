# %%
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
import os.path as osp
import pytorch_lightning as pl
from torch import nn
import numpy as np
from pathlib import Path
from typing import List
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from utils import ImageDatasetFromFile, ToRGB
from reconstruction_model import ReconstructModel
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
project_dir = str(Path(osp.abspath('')).resolve().parents[0])
data_dir = osp.join(project_dir, "data")
model_dir = osp.join(project_dir, "results")

# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)

N_PROTOTYPES = 10

DATASET_CIFAR10 = "cifar10"
DATASET_CIFAR100 = "cifar100"
DATASET_TINYIMAGENET = "tin"
DATASET_IMAGENET200 = "imagenet200"
DATASET_IMAGENET = "imagenet"

i_seed = 0
id_dataset = DATASET_IMAGENET200

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
mean, std = normalization_dict[id_dataset]

# %%

if id_dataset == DATASET_CIFAR10:
    from openood.networks import ResNet18_32x32

    model_paths = [
        osp.join(model_dir, "cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt"),
        osp.join(model_dir, "cifar10_resnet18_32x32_base_e100_lr0.1_default/s1/best.ckpt"),
        osp.join(model_dir, "cifar10_resnet18_32x32_base_e100_lr0.1_default/s2/best.ckpt")
    ]
    backbone = ResNet18_32x32(num_classes=10)
    backbone.load_state_dict(torch.load(model_paths[i_seed]))
    backbone.eval(); backbone.cuda();

    input_dims = [512, 512, 512, 10]
    act_names = ["p4_relu1", "p4_relu1", "p4_relu1", "logit"]
    num_proto = [10, 10, 10, 40]
    model = ReconstructModel(backbone, input_dims, num_proto, act_names).cuda()
    backbone.layer4[1].conv1.register_forward_hook(model.get_activation('p4_relu1'))
    backbone.fc.register_forward_hook(model.get_activation('logit'))

    test_transform = tt.Compose([tt.ToTensor(), tt.Normalize(mean, std)])
    id_train_dataset = ImageDatasetFromFile(
            transform=test_transform,
            txt_file=osp.join(data_dir, "benchmark_imglist/cifar10/train_cifar10.txt"),
            img_dir=osp.join(data_dir, "images_classic")
        )
    batch_size = 32
    augment_resize_vals = [88, 82, 76, 68, 58, 52, 47, 40, 34, 28, 20, 15]

    def get_ood_training_dataset(resize=None):
        transforms = [
            ToRGB(),
            tt.CenterCrop(32),
            tt.ToTensor(),
            tt.Normalize(mean, std)
        ]
        if resize is not None:
            transforms.insert(1, tt.Resize((resize, resize)))
            
        transform = tt.Compose(transforms)
        ood_dataset = ImageDatasetFromFile(
            transform=transform,
            txt_file=osp.join(data_dir, "/benchmark_imglist/cifar10/train_tin597.txt"),
            img_dir=osp.join(data_dir, "images_classic")
        )
        ood_loader = torch.utils.data.DataLoader(
            ood_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        return ood_loader
    
    evaluate_transform = tt.Compose([
        tt.Resize((32, 32)),
        tt.CenterCrop(32),
        tt.ToTensor(),
        tt.Normalize(mean, std)
    ])

elif id_dataset == DATASET_CIFAR100:
    from openood.networks import ResNet18_32x32

    model_paths = [
        osp.join(model_dir, "cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt"),
        osp.join(model_dir, "cifar100_resnet18_32x32_base_e100_lr0.1_default/s1/best.ckpt"),
        osp.join(model_dir, "cifar100_resnet18_32x32_base_e100_lr0.1_default/s2/best.ckpt")
    ]
    backbone = ResNet18_32x32(num_classes=100)
    backbone.load_state_dict(torch.load(model_paths[i_seed]))
    backbone.eval(); backbone.cuda();

    input_dims = [512, 512, 512, 100]
    act_names = ["p4_relu1", "p4_relu1", "p4_relu1", "logit"]
    num_proto = [25, 25, 25, 100]
    model = ReconstructModel(backbone, input_dims, num_proto, act_names).cuda()

    backbone.layer4[1].conv1.register_forward_hook(model.get_activation('p4_relu1'))
    backbone.fc.register_forward_hook(model.get_activation('logit'))

    test_transform = tt.Compose([tt.ToTensor(), tt.Normalize(mean, std)])
    id_train_dataset = ImageDatasetFromFile(
        transform=test_transform,
        txt_file=osp.join(data_dir, "benchmark_imglist/cifar100/train_cifar100.txt"),
        img_dir=osp.join(data_dir, "images_classic")
    )
    batch_size = 16
    augment_resize_vals = [88, 82, 76, 68, 58, 52, 47, 40, 34, 28, 20, 15]

    def get_ood_training_dataset(resize=None):
        transforms = [
            ToRGB(),
            tt.CenterCrop(32),
            tt.ToTensor(),
            tt.Normalize(mean, std)
        ]
        if resize is not None:
            transforms.insert(1, tt.Resize((resize, resize)))
            
        transform = tt.Compose(transforms)
        ood_dataset = ImageDatasetFromFile(
            transform=transform,
            txt_file=osp.join(data_dir, "benchmark_imglist/cifar100/train_tin597.txt"),
            img_dir=osp.join(data_dir, "images_classic")
        )
        ood_loader = torch.utils.data.DataLoader(
            ood_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        return ood_loader
    
    evaluate_transform = tt.Compose([
        tt.Resize((32, 32)),
        tt.CenterCrop(32),
        tt.ToTensor(),
        tt.Normalize(mean, std)
    ])
    
elif id_dataset == DATASET_IMAGENET200:
    from openood.networks import ResNet18_224x224

    model_paths = [
        osp.join(model_dir, "imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best.ckpt"),
        osp.join(model_dir, "imagenet200_resnet18_224x224_base_e90_lr0.1_default/s1/best.ckpt"),
        osp.join(model_dir, "imagenet200_resnet18_224x224_base_e90_lr0.1_default/s2/best.ckpt")
    ]
    backbone = ResNet18_224x224(num_classes=200)
    backbone.load_state_dict(torch.load(model_paths[i_seed]))
    backbone.eval(); backbone.cuda();

    input_dims = [512, 512, 512, 200]
    act_names = ["p4_relu1", "p4_relu1", "p4_relu1", "logit"]
    num_proto = [10, 10, 10, 200]
    model = ReconstructModel(backbone, input_dims, num_proto, act_names).cuda()
    backbone.layer4[1].conv1.register_forward_hook(model.get_activation('p4_relu1'))
    backbone.fc.register_forward_hook(model.get_activation('logit'))

    test_transform = tt.Compose([tt.Resize((224, 224)), tt.ToTensor(), tt.Normalize(mean, std)])
    id_train_dataset = ImageDatasetFromFile(
            transform=test_transform,
            txt_file=osp.join(data_dir, "benchmark_imglist/imagenet200/train_imagenet200.txt"),
            img_dir=osp.join(data_dir, "images_largescale")
        )
    batch_size = 32
    augment_resize_vals = [224-56*2, 224-56, 224+56, 224+56*2]

    evaluate_transform = tt.Compose([
        tt.Resize((224, 224)),
        tt.ToTensor(),
        tt.Normalize(mean, std)
    ])

    def get_ood_training_dataset(resize=None):
        transforms = [
            ToRGB(),
            tt.CenterCrop(224),
            tt.ToTensor(),
            tt.Normalize(mean, std)
        ]
        if resize is not None:
            transforms.insert(1, tt.Resize((resize, resize)))
            
        transform = tt.Compose(transforms)
        ood_dataset = ImageDatasetFromFile(
            transform=transform,
            txt_file=osp.join(data_dir, "/benchmark_imglist/imagenet200/train_imagenet800.txt"),
            img_dir=osp.join(data_dir, "images_largescale")
        )
        ood_loader = torch.utils.data.DataLoader(
            ood_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        return ood_loader


id_train_loader = torch.utils.data.DataLoader(
    id_train_dataset,
    batch_size=batch_size,
    shuffle=True
)
# %%
logger = TensorBoardLogger(save_dir="training_logs")
trainer = pl.Trainer(max_epochs=3, logger=logger, accumulate_grad_batches=1, precision=32, log_every_n_steps=5)
backbone.eval()
trainer.fit(model, id_train_loader)
model.eval(); model.cuda();

# %%
X_id = model.prototype_scores_loader(id_train_loader)

# Get the ood training data
print("Getting the ood training dataset for id classification")
ood_training_loader = get_ood_training_dataset()
X_ood  = model.prototype_scores_loader(ood_training_loader)

#if OOD_TRAIN_DATASET == DATASET_TINYIMAGENET:
for resize in augment_resize_vals:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    print(f"Resize: {resize}")
    ood_training_loader = get_ood_training_dataset(resize=resize)
    X_ood_aug  = model.prototype_scores_loader(ood_training_loader)
    X_ood = np.concatenate([X_ood, X_ood_aug], axis=0)

# %%

# Gather the data for OOD classifer training
y_ood = np.zeros(len(X_ood), dtype=np.int32)
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
from mlp import MLP, OODClassiferDataset

# Train the OOD classifier
ood_classifer = MLP(input_size=X_train.shape[1]).cuda()
ood_classifer_dataset = OODClassiferDataset(X_train, y_arr)
ood_classifer_loader = torch.utils.data.DataLoader(
    ood_classifer_dataset,
    batch_size=100,
    shuffle=True
)

# Train the model
logger = TensorBoardLogger(save_dir="training_logs_metamodel")
trainer = pl.Trainer(max_epochs=2, logger=logger, accumulate_grad_batches=1, precision=32, log_every_n_steps=5)
trainer.fit(ood_classifer, ood_classifer_loader)
ood_classifer.eval(); ood_classifer.cuda();

# %%

# Make predictions on the test set
with torch.no_grad():
    ood_classifer_loader_noshuffle = torch.utils.data.DataLoader(
        ood_classifer_dataset,
        batch_size=1000, 
        shuffle=False
    )
    preds = [] 
    for x_batch, _ in tqdm(ood_classifer_loader_noshuffle):
        preds.append(ood_classifer(x_batch.cuda()).cpu())
    
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
from utils import PrototypePostprocessor
from openood.evaluation_api import Evaluator

# Initialize an evaluator and evaluate
evaluator = Evaluator(model, id_name=id_dataset,
    preprocessor=evaluate_transform, postprocessor=PrototypePostprocessor(None, scaler, ood_classifer), batch_size=50)
metrics = evaluator.eval_ood()
# %%
