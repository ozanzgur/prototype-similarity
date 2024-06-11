# Before running this script, download one or all parts of the imagenet1k dataset, place them under data/images_largescale/imagenet_1k/train
# Download from: https://huggingface.co/datasets/ILSVRC/imagenet-1k/tree/main/data
# or comment out the imagenet200 runs

# Download all datasets to data/ (except imagenet_1k training set), checkpoints to results/
#python download_data.py --contents datasets --datasets default
#python download_data.py --contents datasets --datasets "ood_v1.5"
#python download_data.py --contents checkpoints
# download_data.py was copied from: https://github.com/Jingkang50/OpenOOD/tree/main/scripts/download

# default settings
python prototype_similarity/train_test.py --id_dataset cifar10 --i_seed 0
python prototype_similarity/train_test.py --id_dataset cifar10 --i_seed 1
python prototype_similarity/train_test.py --id_dataset cifar10 --i_seed 2

python prototype_similarity/train_test.py --id_dataset cifar100 --i_seed 0
python prototype_similarity/train_test.py --id_dataset cifar100 --i_seed 1
python prototype_similarity/train_test.py --id_dataset cifar100 --i_seed 2

python prototype_similarity/train_test.py --id_dataset imagenet200 --i_seed 0 --do_augment True --spatial_avg_features False --train_prototypes True --dropout_rate 0.8 --mlp_hidden_size 250 --ood_train_size 50000 --id_train_size 50000
python prototype_similarity/train_test.py --id_dataset imagenet200 --i_seed 1 --do_augment True --spatial_avg_features False --train_prototypes True --dropout_rate 0.8 --mlp_hidden_size 250 --ood_train_size 50000 --id_train_size 50000
python prototype_similarity/train_test.py --id_dataset imagenet200 --i_seed 2 --do_augment True --spatial_avg_features False --train_prototypes True --dropout_rate 0.8 --mlp_hidden_size 250 --ood_train_size 50000 --id_train_size 50000

python prototype_similarity/train_test.py --id_dataset imagenet200 --i_seed 0 --fsood True --resize_augmentation False --spatial_avg_features False
python prototype_similarity/train_test.py --id_dataset imagenet200 --i_seed 1 --fsood True --resize_augmentation False --spatial_avg_features False
python prototype_similarity/train_test.py --id_dataset imagenet200 --i_seed 2 --fsood True --resize_augmentation False --spatial_avg_features False