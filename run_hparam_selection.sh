id_dataset="cifar10",
    i_seed=0,
    do_augment=True,
    ood_train_size=None, # None: Use all available data
    prototype_layer_name=None, # None: Use default prototypes
    prototype_channels=10,
    prototype_count=10,
    spatial_avg_features=False,
    mlp_input_size=4880,
    dropout_rate=0.5,
    mlp_hidden_size=250,
    train_prototypes=True,
    mlp_lr=1e-3,
    fsood=False

# CIFAR10 ##########################################################################################################################
# Augmentation
# Lower val loss without augmentation
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar10 --do_augment False --spatial_avg_features False --dropout_rate 0.8 --mlp_hidden_size 250
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar10 --do_augment True --spatial_avg_features False --dropout_rate 0.8 --mlp_hidden_size 250

# Train prototypes
# Lower val loss without training the prototypes
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar10 --train_prototypes False --do_augment False --spatial_avg_features False --dropout_rate 0.8 --mlp_hidden_size 250
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar10 --train_prototypes False --do_augment True --spatial_avg_features False --dropout_rate 0.8 --mlp_hidden_size 250

# MLP hidden size
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar10 --do_augment False --mlp_hidden_size 10 --spatial_avg_features False --dropout_rate 0.8
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar10 --do_augment False --mlp_hidden_size 50 --spatial_avg_features False --dropout_rate 0.8
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar10 --do_augment False --mlp_hidden_size 100 --spatial_avg_features False --dropout_rate 0.8
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar10 --do_augment False --mlp_hidden_size 250 --spatial_avg_features False --dropout_rate 0.8
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar10 --do_augment False --mlp_hidden_size 500 --spatial_avg_features False --dropout_rate 0.8

# Dropout rate
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar10 --do_augment False --dropout_rate 0.2 --mlp_hidden_size 250 --spatial_avg_features False
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar10 --do_augment False --dropout_rate 0.5 --mlp_hidden_size 250 --spatial_avg_features False
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar10 --do_augment False --dropout_rate 0.8 --mlp_hidden_size 250 --spatial_avg_features False
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar10 --do_augment False --dropout_rate 0.9 --mlp_hidden_size 250 --spatial_avg_features False

# spatial_avg_feature
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar10 --do_augment False --spatial_avg_features True
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar10 --do_augment False --spatial_avg_features False

# CIFAR100 #########################################################################################################################
# Augmentation
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar100 --do_augment True --spatial_avg_features False --dropout_rate 0.8 --mlp_hidden_size 250 --mlp_input_size 5000
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar100 --do_augment False --spatial_avg_features False --dropout_rate 0.8 --mlp_hidden_size 250 --mlp_input_size 5000

# Train prototypes
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar100 --train_prototypes False --do_augment False --spatial_avg_features False --dropout_rate 0.8 --mlp_hidden_size 250
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar100 --train_prototypes False --do_augment True --spatial_avg_features False --dropout_rate 0.8 --mlp_hidden_size 250

# MLP hidden size
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar100 --do_augment False --mlp_hidden_size 10 --spatial_avg_features False --dropout_rate 0.8
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar100 --do_augment False --mlp_hidden_size 50 --spatial_avg_features False --dropout_rate 0.8
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar100 --do_augment False --mlp_hidden_size 100 --spatial_avg_features False --dropout_rate 0.8
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar100 --do_augment False --mlp_hidden_size 250 --spatial_avg_features False --dropout_rate 0.8
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar100 --do_augment False --mlp_hidden_size 500 --spatial_avg_features False --dropout_rate 0.8

# Dropout rate
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar100 --do_augment False --dropout_rate 0.2 --mlp_hidden_size 250 --spatial_avg_features False
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar100 --do_augment False --dropout_rate 0.5 --mlp_hidden_size 250 --spatial_avg_features False
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar100 --do_augment False --dropout_rate 0.8 --mlp_hidden_size 250 --spatial_avg_features False
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar100 --do_augment False --dropout_rate 0.9 --mlp_hidden_size 250 --spatial_avg_features False

# spatial_avg_feature
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar100 --do_augment False --spatial_avg_features True
python prototype_similarity/train_reconstruction_nn_pt.py --id_dataset cifar100 --do_augment False --spatial_avg_features False


