# %%
import torch
import numpy as np
import torchvision.transforms as tt
import random
import os.path as osp
import os
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import fire
import pandas as pd
from openood.evaluation_api import Evaluator
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import PrototypePostprocessor, subsample_dataset, MergedDataset, plot_prot_usage, plot_prot_corrs, plot_prot_weights

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from mlp import MLP
from utils import ImageDatasetFromFile, ToRGB
from reconstruction_model import ReconstructModel

project_dir = str(Path(__file__).resolve().parents[1])
data_dir = osp.join(project_dir, "data")
model_dir = osp.join(project_dir, "results")

def set_seed(seed):
    random.seed(seed)  # Set seed for the built-in random module
    np.random.seed(seed)  # Set seed for NumPy
    torch.manual_seed(seed)  # Set seed for PyTorch CPU
    torch.cuda.manual_seed(seed)  # Set seed for PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs

    # Ensure that the CUDNN operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Example usage
set_seed(42)


# train_tin597: 29850
# cifar100 train: 50000

DATASET_CIFAR10 = "cifar10"
DATASET_CIFAR100 = "cifar100"
DATASET_TINYIMAGENET = "tin"
DATASET_IMAGENET200 = "imagenet200"
DATASET_IMAGENET = "imagenet"
default_id_dataset = DATASET_CIFAR10
default_prot_train_epochs = 3
default_reconstruct_output_act_size = 4

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

def main(
    id_dataset="cifar10",
    i_seed=0,
    do_augment=True,
    ood_train_size=None, # None: Use all available data
    id_train_size=None, # None: Use all available data
    prototype_layer_name=None, # None: Use default prototypes
    prototype_channels=10,
    prototype_count=10,
    spatial_avg_features=False,
    dropout_rate=0.8,
    mlp_hidden_size=250,
    train_prototypes=True,
    mlp_lr=1e-3,
    fsood=False,
    do_plot_prot_usage=False, # Heatmap of prototype usage counts in reconstruction during training
    reconstruct_output_act_size=default_reconstruct_output_act_size,
    is_conv_method=False,
    prot_train_epochs=default_prot_train_epochs,
    measure_feature_importance=False,
    save_act_samples=False
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
            num_proto = [100, 100, 100, 100]
            #input_dims = [512]
            #act_names = ["p4_relu1"]
            #num_proto = [20]
        model = ReconstructModel(backbone, input_dims, num_proto, act_names, 
                                 spatial_avg_features, reconstruct_output_act_size, is_conv_method=is_conv_method, model_name=id_dataset).cuda()
        if prototype_layer_name:
            get_resnet_layer(backbone, prototype_layer_name).register_forward_hook(model.get_activation(prototype_layer_name))

        else:
            backbone.layer2[1].conv1.register_forward_hook(model.get_activation('p2_relu1'))
            backbone.layer3[1].conv1.register_forward_hook(model.get_activation('p3_relu1'))
            backbone.layer4[1].conv1.register_forward_hook(model.get_activation('p4_relu1'))
            backbone.fc.register_forward_hook(model.get_activation('logit'))

        batch_size = 25

        def get_id_dataset(split="train", augmentation=False):
            if augmentation:
                id_transform = tt.Compose([
                    ToRGB(),
                    tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    tt.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
                    tt.RandomApply([tt.Resize((32, 32))], p=1.0),
                    tt.CenterCrop(32),
                    tt.ToTensor(), 
                    tt.Normalize(mean, std)
                ])
            else:
                id_transform = tt.Compose([ToRGB(), tt.ToTensor(), tt.Normalize(mean, std)])
            id_dataset = ImageDatasetFromFile(
                    transform=id_transform,
                    txt_file=osp.join(data_dir, f"benchmark_imglist/cifar10/{split}_cifar10.txt"),
                    img_dir=osp.join(data_dir, "images_classic")
            )
            return id_dataset
        
        def get_ood_dataset(split="train", augmentation=False):
            if augmentation:
                transforms = [
                    ToRGB(),
                    #tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    tt.RandomAffine(degrees=0, translate=(0.0, 0.0), scale=(0.25, 1.3), shear=0),
                    tt.RandomApply([tt.Resize((32, 32))], p=0.5),
                    tt.CenterCrop(32),
                    tt.ToTensor(),
                    tt.Normalize(mean, std)
                ]
            else:
                transforms = [
                    ToRGB(),
                    tt.Resize((32, 32)),
                    tt.CenterCrop(32),
                    tt.ToTensor(),
                    tt.Normalize(mean, std)
                ]
                
            dataset_filename = "train_tin597.txt" if split == "train" else "val_tin.txt"
            transform = tt.Compose(transforms)
            ood_dataset = ImageDatasetFromFile(
                transform=transform,
                txt_file=osp.join(data_dir, f"benchmark_imglist/cifar10/{dataset_filename}"),
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
            input_dims = [128, 256, 512, 512, 100]
            act_names = ["p2_relu1", "p3_relu1", "p4_relu1", "avgpool", "logit"]
            num_proto = [100, 100, 100, 100, 100]
        model = ReconstructModel(backbone, input_dims, num_proto, act_names, 
                                 spatial_avg_features, reconstruct_output_act_size, is_conv_method=is_conv_method, model_name=id_dataset).cuda()

        if prototype_layer_name:
            get_resnet_layer(backbone, prototype_layer_name).register_forward_hook(model.get_activation(prototype_layer_name))
        else:
            backbone.layer2[1].conv1.register_forward_hook(model.get_activation('p2_relu1'))
            backbone.layer3[1].conv1.register_forward_hook(model.get_activation('p3_relu1'))
            backbone.layer4[1].conv1.register_forward_hook(model.get_activation('p4_relu1'))
            backbone.avgpool.register_forward_hook(model.get_activation('avgpool'))
            backbone.fc.register_forward_hook(model.get_activation('logit'))

        batch_size = 25

        def get_id_dataset(split="train"):
            id_transform = tt.Compose([ToRGB(), tt.Resize((32, 32)), tt.ToTensor(), tt.Normalize(mean, std)])
            id_dataset = ImageDatasetFromFile(
                    transform=id_transform,
                    txt_file=osp.join(data_dir, f"benchmark_imglist/cifar100/{split}_cifar100.txt"),
                    img_dir=osp.join(data_dir, "images_classic")
            )
            return id_dataset
        

        def get_ood_dataset(split="train", augmentation=False):
            if augmentation:
                transforms = [
                    ToRGB(),
                    #tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    tt.RandomAffine(degrees=0, translate=(0.0, 0.0), scale=(0.25, 1.3), shear=0),
                    tt.RandomApply([tt.Resize((32, 32))], p=0.5),
                    tt.CenterCrop(32),
                    tt.ToTensor(),
                    tt.Normalize(mean, std)
                ]
            else:
                transforms = [
                    ToRGB(),
                    tt.Resize((32, 32)),
                    tt.CenterCrop(32),
                    tt.ToTensor(),
                    tt.Normalize(mean, std)
                ]
            
            transform = tt.Compose(transforms)
            dataset_filename = "train_tin597.txt" if split == "train" else "val_tin.txt"
            ood_dataset = ImageDatasetFromFile(
                transform=transform,
                txt_file=osp.join(data_dir, f"benchmark_imglist/cifar100/{dataset_filename}"),
                img_dir=osp.join(data_dir, "images_classic")
            )
            if ood_train_size:
                ood_dataset = subsample_dataset(ood_dataset, ood_train_size)
            return ood_dataset
        
        evaluate_transform = tt.Compose([
            ToRGB(),
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
            # input_dims = [256, 512] # , 200
            # act_names = ["p3_relu1", "avgpool"] # , "logit"
            # num_proto = [100, 100] # 
            input_dims = [256, 200] # , 200
            act_names = ["p3_relu1", "logit"] # , "logit"
            num_proto = [100, 5] # 
        model = ReconstructModel(backbone, input_dims, num_proto, act_names, 
                                 spatial_avg_features, reconstruct_output_act_size, 
                                 is_conv_method=is_conv_method, model_name=id_dataset).cuda()
        if prototype_layer_name:
            get_resnet_layer(backbone, prototype_layer_name).register_forward_hook(model.get_activation(prototype_layer_name))
        else:
            backbone.layer1[1].conv1.register_forward_hook(model.get_activation('p1_relu1'))
            backbone.layer2[1].conv1.register_forward_hook(model.get_activation('p2_relu1'))
            backbone.layer3[1].conv1.register_forward_hook(model.get_activation('p3_relu1'))
            backbone.layer4[1].conv1.register_forward_hook(model.get_activation('p4_relu1'))
            backbone.avgpool.register_forward_hook(model.get_activation('avgpool'))
            backbone.fc.register_forward_hook(model.get_activation('logit'))

        def get_id_dataset(split="train", augmentation=False):
            if augmentation:
                transforms = [
                    ToRGB(),
                    tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    tt.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.75, 1.25), shear=10),
                    tt.RandomApply([tt.Resize((224, 224))], p=0.3),
                    tt.CenterCrop(224),
                    tt.ToTensor(),
                    tt.Normalize(mean, std)
                ]
            else:
                transforms = [
                    ToRGB(), tt.Resize((224, 224)), tt.ToTensor(), tt.Normalize(mean, std)
                ]

            id_transform = tt.Compose(transforms)
            id_dataset = ImageDatasetFromFile(
                    transform=id_transform,
                    txt_file=osp.join(data_dir, f"benchmark_imglist/imagenet200/{split}_imagenet200.txt"),
                    img_dir=osp.join(data_dir, "images_largescale"),
                    is_imagenet=(split=="train")
            )
            if id_train_size:
                id_dataset = subsample_dataset(id_dataset, id_train_size)
            return id_dataset

        batch_size = 16

        evaluate_transform = tt.Compose([
            ToRGB(),
            tt.Resize((224, 224)),
            tt.ToTensor(),
            tt.Normalize(mean, std)
        ])

        def get_ood_dataset(split="train", augmentation=False):
            if augmentation:
                transforms = [
                    ToRGB(),
                    #tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    tt.RandomAffine(degrees=0, translate=(0.0, 0.0), scale=(0.7, 1.3), shear=0),
                    tt.RandomApply([tt.Resize((224, 224))], p=0.5),
                    tt.CenterCrop(224),
                    tt.ToTensor(),
                    tt.Normalize(mean, std)
                ]
            else:
                transforms = [
                    ToRGB(),
                    tt.Resize((224, 224)),
                    tt.CenterCrop(224),
                    tt.ToTensor(),
                    tt.Normalize(mean, std)
                ]
                
            transform = tt.Compose(transforms)
            dataset_filename = "train_imagenet800.txt" if split == "train" else "val_openimage_o.txt"
            ood_dataset = ImageDatasetFromFile(
                transform=transform,
                txt_file=osp.join(data_dir, f"benchmark_imglist/imagenet200/{dataset_filename}"),
                img_dir=osp.join(data_dir, "images_largescale"),
                is_imagenet=(split == "train")
            )
            if ood_train_size:
                ood_dataset = subsample_dataset(ood_dataset, ood_train_size)
            return ood_dataset


    id_train_dataset = get_id_dataset(split="train")
    id_train_loader = torch.utils.data.DataLoader(
        id_train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Train prototypes
    if not is_conv_method:
        logger = TensorBoardLogger(save_dir="training_logs")
        if train_prototypes:
            # Initialize with the same std as activations
            acts = model.get_all_acts(next(iter(id_train_loader))[0].cuda())
            for act, matcher in zip(acts, model.prototype_matchers):
                matcher.init_with_acts(act)

            trainer = pl.Trainer(max_epochs=prot_train_epochs, logger=logger, accumulate_grad_batches=100, precision=32, log_every_n_steps=5)
            backbone.eval()

            # Save the first batch activations
            if save_act_samples:
                model.save_tag = "train_prototypes"
                model.save_act_samples = True
                model.reset_save_counter()

            trainer.fit(model, id_train_loader)
            model.save_act_samples = False
        model.eval(); model.cuda();

        if do_plot_prot_usage:
            plot_prot_usage(model)
            plot_prot_corrs(model)
            plot_prot_weights(model)

    ood_training_dataset = get_ood_dataset(split="train", augmentation=do_augment)
    mixed_train_dataset = MergedDataset(ood_training_dataset, id_train_dataset)
    ood_classifer_train_loader = torch.utils.data.DataLoader(
        mixed_train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    ood_val_dataset = get_ood_dataset(split="val")
    id_val_dataset = get_id_dataset(split="val")
    mixed_val_dataset = MergedDataset(ood_val_dataset, id_val_dataset)
    ood_classifer_val_loader = torch.utils.data.DataLoader(
        mixed_val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Train OOD classifier
    model.disable_prototype_training()
    mlp_input_size = model.prototype_scores(next(iter(ood_classifer_train_loader))[0].cuda()).cpu().detach().shape[1]
    print("mlp_hidden_size", mlp_hidden_size)
    ood_classifer = MLP(reconstructor=model, input_size=mlp_input_size, hidden_size=mlp_hidden_size, learning_rate=mlp_lr, dropout_rate=dropout_rate).cuda()

    # Save metrics
    metrics_filename = f"{id_dataset}_s-{i_seed}_augment-{do_augment}_oodsize-{ood_train_size if ood_train_size else 'all'}_avg-{spatial_avg_features}_dr-{dropout_rate}_mlpsize-{mlp_hidden_size}_trn_prot-{train_prototypes}"
    if fsood:
        metrics_filename = metrics_filename + "_fsood"
    if prototype_layer_name:
        metrics_filename = metrics_filename + f"_protlayer-{prototype_layer_name}_protch-{prototype_channels}_nprot-{prototype_count}"
    if is_conv_method:
        metrics_filename = metrics_filename + "_conv1x1"
    if prot_train_epochs != default_prot_train_epochs:
        metrics_filename = metrics_filename + f"_prot-ep-{prot_train_epochs}"

    if reconstruct_output_act_size != default_reconstruct_output_act_size:
        metrics_filename = metrics_filename + f"_act-size-{reconstruct_output_act_size}"


    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min'
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        #dirpath=f'{project_dir}/models/{metrics_filename}',
        filename='ep-{epoch:02d}_loss-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
        save_weights_only=True
    )

    if save_act_samples:
        model.save_act_samples = True
        model.save_tag = "train_detector"
        model.reset_save_counter()

    # Train the model
    logger = TensorBoardLogger(save_dir="training_logs_metamodel", version=metrics_filename)
    trainer = pl.Trainer(callbacks=[early_stop_callback, checkpoint_callback], max_epochs=5,
                        logger=logger, accumulate_grad_batches=100, precision=32, log_every_n_steps=1)
    trainer.fit(ood_classifer, ood_classifer_train_loader, ood_classifer_val_loader)
    print(f"Loading the best model from: {checkpoint_callback.best_model_path}")
    ood_classifer = MLP.load_from_checkpoint(checkpoint_callback.best_model_path, reconstructor=model, input_size=mlp_input_size, hidden_size=mlp_hidden_size)
    ood_classifer.eval(); ood_classifer.cuda();

    model.save_act_samples = False

    # Feature importance
    if measure_feature_importance:
        # Initialize an evaluator and evaluate
        evaluator = Evaluator(model, id_name=id_dataset,
        preprocessor=evaluate_transform, postprocessor=PrototypePostprocessor(None, ood_classifer), batch_size=batch_size)
        metrics_all_features = evaluator.eval_ood(fsood=fsood)
        all_auroc_near = metrics_all_features.loc[metrics_all_features.index == "nearood", "AUROC"].iloc[0]
        all_auroc_far = metrics_all_features.loc[metrics_all_features.index == "farood", "AUROC"].iloc[0]
        all_fpr_near = metrics_all_features.loc[metrics_all_features.index == "nearood", "FPR@95"].iloc[0]
        all_fpr_far = metrics_all_features.loc[metrics_all_features.index == "farood", "FPR@95"].iloc[0]

        imp_metrics = []
        model.feature_masking = True
        model.mask_layer_i = 0
        for i_prot in range(num_proto[0]):
            print(f"Feature Importance, Prototype: {i_prot}/{num_proto[0]} #######################")
            model.mask_prot_i = i_prot
            # Initialize an evaluator and evaluate
            evaluator = Evaluator(model, id_name=id_dataset,
                preprocessor=evaluate_transform, postprocessor=PrototypePostprocessor(None, ood_classifer), batch_size=batch_size)
            metrics = evaluator.eval_ood(fsood=fsood)
            imp_metrics.append((
                i_prot,
                metrics.loc[metrics.index == "nearood", "AUROC"].iloc[0] - all_auroc_near,
                metrics.loc[metrics.index == "nearood", "FPR@95"].iloc[0] - all_fpr_near,
                metrics.loc[metrics.index == "farood", "AUROC"].iloc[0] - all_auroc_far,
                metrics.loc[metrics.index == "farood", "FPR@95"].iloc[0] - all_fpr_far
            ))
        imp_metrics = pd.DataFrame(imp_metrics, columns=["prot_i", "AUROC-nearood", "FPR95-nearood", "AUROC-farood", "FPR95-farood"])
        metrics_dir = osp.join(project_dir, "metrics")
        imp_metrics = imp_metrics.sort_values(by="AUROC-nearood")
        imp_metrics.to_csv(osp.join(metrics_dir, "feature_importance.csv"), index=False)

        near_ood_order = imp_metrics["prot_i"].values
        print("Feature importances:")
        print(imp_metrics)
        plot_prot_weights(model, order=near_ood_order)
        plot_prot_corrs(model, order=near_ood_order)

    else:
        # Save the first batch activations
        if save_act_samples:
            model.save_act_samples = True
            model.save_tag = "eval"
            model.reset_save_counter()

        # Initialize an evaluator and evaluate
        evaluator = Evaluator(model, id_name=id_dataset,
            preprocessor=evaluate_transform, postprocessor=PrototypePostprocessor(None, ood_classifer), batch_size=batch_size)
        metrics = evaluator.eval_ood(fsood=fsood)

        metrics = pd.DataFrame(metrics)
        metrics_dir = osp.join(project_dir, "metrics")
        if not osp.exists(metrics_dir):
            os.mkdir(metrics_dir)

        metrics.to_csv(osp.join(metrics_dir, metrics_filename + ".csv"))

if __name__ == '__main__':
   fire.Fire(main)
# %%
"""
main(id_dataset="cifar100", prototype_count=20, prototype_channels=512, prototype_layer_name="3_1_conv1", measure_feature_importance=True)
"""