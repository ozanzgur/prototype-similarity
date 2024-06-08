# Before running this script, download one or all parts of the imagenet1k dataset, place them under data/images_largescale/imagenet_1k/train
# Download from: https://huggingface.co/datasets/ILSVRC/imagenet-1k/tree/main/data
# or comment out the imagenet200 runs

# Download all datasets to data/ (except imagenet_1k training set), checkpoints to results/
#python download_data.py --contents datasets --datasets default
#python download_data.py --contents datasets --datasets "ood_v1.5"
#python download_data.py --contents checkpoints
# download_data.py was copied from: https://github.com/Jingkang50/OpenOOD/tree/main/scripts/download

# default settings
python prototype_similarity/train_reconstruction_nn.py --id_dataset cifar10 --i_seed 0 --resize_augmentation True --spatial_avg_features False
python prototype_similarity/train_reconstruction_nn.py --id_dataset cifar10 --i_seed 1 --resize_augmentation True --spatial_avg_features False
python prototype_similarity/train_reconstruction_nn.py --id_dataset cifar10 --i_seed 2 --resize_augmentation True --spatial_avg_features False
python prototype_similarity/train_reconstruction_nn.py --id_dataset cifar100 --i_seed 0 --resize_augmentation True --spatial_avg_features False
python prototype_similarity/train_reconstruction_nn.py --id_dataset cifar100 --i_seed 1 --resize_augmentation True --spatial_avg_features False
python prototype_similarity/train_reconstruction_nn.py --id_dataset cifar100 --i_seed 2 --resize_augmentation True --spatial_avg_features False
python prototype_similarity/train_reconstruction_nn.py --id_dataset imagenet200 --i_seed 0 --resize_augmentation False --spatial_avg_features False
python prototype_similarity/train_reconstruction_nn.py --id_dataset imagenet200 --i_seed 1 --resize_augmentation False --spatial_avg_features False
python prototype_similarity/train_reconstruction_nn.py --id_dataset imagenet200 --i_seed 2 --resize_augmentation False --spatial_avg_features False
python prototype_similarity/train_reconstruction_nn.py --id_dataset imagenet200 --i_seed 0 --fsood True --resize_augmentation False --spatial_avg_features False
python prototype_similarity/train_reconstruction_nn.py --id_dataset imagenet200 --i_seed 1 --fsood True --resize_augmentation False --spatial_avg_features False
python prototype_similarity/train_reconstruction_nn.py --id_dataset imagenet200 --i_seed 2 --fsood True --resize_augmentation False --spatial_avg_features False