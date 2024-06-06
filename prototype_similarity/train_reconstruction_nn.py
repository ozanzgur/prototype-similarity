# %%
import torch
import numpy as np
import torchvision.transforms as tt
import random
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm
import os.path as osp
import os
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import fire
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from openood.evaluation_api import Evaluator
from utils import PrototypePostprocessor, subsample_dataset

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from mlp import MLP, OODClassiferDataset
from utils import ImageDatasetFromFile, ToRGB
from reconstruction_model import ReconstructModel

project_dir = str(Path(__file__).resolve().parents[1])
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
default_id_dataset = DATASET_IMAGENET200

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

def get_resnet_layer(model, layer_name):
    if layer_name == "conv1":
        return model.conv1
    if layer_name == "bn1":
        return model.bn1
    if layer_name == "fc":
        return model.fc
    if layer_name == "penultimate":
        return model.avgpool

    i_layer, i_block, layer_type = layer_name.split('_')
    layer = getattr(model, f"layer{int(i_layer)+1}")
    block = layer[int(i_block)]
    layer = getattr(block, layer_type)
    return layer

"""id_dataset=default_id_dataset,
        i_seed=0,
        resize_augmentation=True,
        ood_train_size=None, # None: Use all available data
        prototype_layer_name=None, # None: Use default prototypes
        prototype_channels=10,
        prototype_count=10,
        spatial_avg_features=False"""

def main(
        id_dataset="cifar10",
        i_seed=0,
        resize_augmentation=False,
        ood_train_size=None, # None: Use all available data
        prototype_layer_name=None, # None: Use default prototypes
        prototype_channels=10,
        prototype_count=10,
        spatial_avg_features=False,
        fsood=False
        ):
    assert id_dataset in [DATASET_CIFAR10, DATASET_CIFAR100, DATASET_IMAGENET200]
    mean, std = normalization_dict[id_dataset]
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

        if prototype_layer_name:
            input_dims = [prototype_channels]
            act_names = [prototype_layer_name]
            num_proto = [prototype_count]

        else:
            input_dims = [128, 256, 512, 10]
            act_names = ["p2_relu1", "p3_relu1", "p4_relu1", "logit"]
            num_proto = [10, 10, 10, 40]
        model = ReconstructModel(backbone, input_dims, num_proto, act_names, spatial_avg_features).cuda()
        if prototype_layer_name:
            get_resnet_layer(backbone, prototype_layer_name).register_forward_hook(model.get_activation(prototype_layer_name))

        else:
            backbone.layer2[1].conv1.register_forward_hook(model.get_activation('p2_relu1'))
            backbone.layer3[1].conv1.register_forward_hook(model.get_activation('p3_relu1'))
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
                txt_file=osp.join(data_dir, "benchmark_imglist/cifar10/train_tin597.txt"),
                img_dir=osp.join(data_dir, "images_classic")
            )
            if ood_train_size:
                ood_dataset = subsample_dataset(ood_dataset, ood_train_size)
            return ood_dataset
        
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

        if prototype_layer_name:
            input_dims = [prototype_channels]
            act_names = [prototype_layer_name]
            num_proto = [prototype_count]

        else:
            input_dims = [128, 256, 512, 100]
            act_names = ["p2_relu1", "p3_relu1", "p4_relu1", "logit"]
            num_proto = [25, 25, 25, 100]
        model = ReconstructModel(backbone, input_dims, num_proto, act_names, spatial_avg_features).cuda()
        if prototype_layer_name:
            get_resnet_layer(backbone, prototype_layer_name).register_forward_hook(model.get_activation(prototype_layer_name))
        else:
            backbone.layer2[1].conv1.register_forward_hook(model.get_activation('p2_relu1'))
            backbone.layer3[1].conv1.register_forward_hook(model.get_activation('p3_relu1'))
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
            if ood_train_size:
                ood_dataset = subsample_dataset(ood_dataset, ood_train_size)
            return ood_dataset
        
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

        if prototype_layer_name:
            input_dims = [prototype_channels]
            act_names = [prototype_layer_name]
            num_proto = [prototype_count]

        else:
            input_dims = [256, 512, 200]
            act_names = ["p3_relu1", "p4_relu1", "logit"]
            num_proto = [10, 25, 50]
        model = ReconstructModel(backbone, input_dims, num_proto, act_names, spatial_avg_features).cuda()
        if prototype_layer_name:
            get_resnet_layer(backbone, prototype_layer_name).register_forward_hook(model.get_activation(prototype_layer_name))
        else:
            backbone.layer2[1].conv1.register_forward_hook(model.get_activation('p2_relu1'))
            backbone.layer3[1].conv1.register_forward_hook(model.get_activation('p3_relu1'))
            backbone.layer4[1].conv1.register_forward_hook(model.get_activation('p4_relu1'))
            backbone.avgpool.register_forward_hook(model.get_activation('avgpool'))
            backbone.fc.register_forward_hook(model.get_activation('logit'))

        test_transform = tt.Compose([ToRGB(), tt.Resize((224, 224)), tt.ToTensor(), tt.Normalize(mean, std)])
        id_train_dataset = ImageDatasetFromFile(
                transform=test_transform,
                txt_file=osp.join(data_dir, "benchmark_imglist/imagenet200/train_imagenet200.txt"),
                img_dir=osp.join(data_dir, "images_largescale"),
                is_imagenet=True
            )
        batch_size = 32
        augment_resize_vals = [224-56*2, 224-56, 224+56, 224+56*2, 224+56*3]

        evaluate_transform = tt.Compose([
            ToRGB(),
            tt.Resize((224, 224)),
            tt.ToTensor(),
            tt.Normalize(mean, std)
        ])

        def get_ood_training_dataset(resize=None):
            transforms = [
                ToRGB(),
                tt.CenterCrop(224),
                tt.Resize((224, 224)),
                tt.ToTensor(),
                tt.Normalize(mean, std)
            ]
            if resize is not None:
                transforms.insert(1, tt.Resize((resize, resize)))
                
            transform = tt.Compose(transforms)
            ood_dataset = ImageDatasetFromFile(
                transform=transform,
                txt_file=osp.join(data_dir, "benchmark_imglist/imagenet200/train_imagenet800.txt"),
                img_dir=osp.join(data_dir, "images_largescale"),
                is_imagenet=True
            )
            if ood_train_size:
                ood_dataset = subsample_dataset(ood_dataset, ood_train_size)
            return ood_dataset


    id_train_loader = torch.utils.data.DataLoader(
        id_train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    logger = TensorBoardLogger(save_dir="training_logs")
    trainer = pl.Trainer(max_epochs=3, logger=logger, accumulate_grad_batches=1, precision=32, log_every_n_steps=5)
    backbone.eval()
    trainer.fit(model, id_train_loader)
    model.eval(); model.cuda();

    X_id = model.prototype_scores_loader(id_train_loader)

    # Get the ood training data
    print("Getting the ood training dataset for id classification")
    
    ood_training_dataset = get_ood_training_dataset()
    ood_train_loader = torch.utils.data.DataLoader(
        ood_training_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    X_ood  = model.prototype_scores_loader(ood_train_loader)

    print(f"resize_augmentation enabled: {resize_augmentation}")
    if resize_augmentation:
        for resize in augment_resize_vals:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
            print(f"Resize: {resize}")
            ood_training_dataset = get_ood_training_dataset(resize=resize)
            ood_train_loader = torch.utils.data.DataLoader(
                ood_training_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            X_ood_aug  = model.prototype_scores_loader(ood_train_loader)
            X_ood = np.concatenate([X_ood, X_ood_aug], axis=0)

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

    # Train the OOD classifier
    ood_classifer = MLP(input_size=X_train.shape[1], hidden_size=250).cuda()
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


    # Initialize an evaluator and evaluate
    evaluator = Evaluator(model, id_name=id_dataset,
        preprocessor=evaluate_transform, postprocessor=PrototypePostprocessor(None, scaler, ood_classifer), batch_size=batch_size)
    metrics = evaluator.eval_ood(fsood=fsood)

    metrics = pd.DataFrame(metrics)
    metrics_dir = osp.join(project_dir, "metrics")
    if not osp.exists(metrics_dir):
        os.mkdir(metrics_dir)

    # Save metrics
    metrics_filename = f"{id_dataset}_s-{i_seed}_resize-{resize_augmentation}_oodsize-{ood_train_size if ood_train_size else 'all'}_avg-{spatial_avg_features}"
    if fsood:
        metrics_filename = metrics_filename + "_fsood"
    if prototype_layer_name:
        metrics_filename = metrics_filename + f"_protlayer-{prototype_layer_name}_protch-{prototype_channels}_nprot-{prototype_count}"
    metrics.to_csv(osp.join(metrics_dir, metrics_filename + ".csv"))

# %%
if __name__ == '__main__':
    fire.Fire(main)