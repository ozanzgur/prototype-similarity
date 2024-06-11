# Prototype Similarity
Repository for the out-of-distribution(OOD) detection method named Prototype Similarity

## Metrics
Metrics for CIFAR10, CIFAR100, ImageNet-200, ImageNet-200-FS can be found in metrics/

## Reproduce
- Run script_data_download.sh
- Run script_reproduce.sh

Reproduction script trains the OOD detection models on CIFAR10, CIFAR100, ImageNet-200 datasets, then tests them on OpenOOD benchmarks.
download_data.py script was copied from: https://github.com/Jingkang50/OpenOOD/blob/main/scripts/download/download.py

Before running script_reproduce.sh, download one or all parts of the imagenet1k dataset, place them under data/images_largescale/imagenet_1k/train

Download from: https://huggingface.co/datasets/ILSVRC/imagenet-1k/tree/main/data



