""" Run script_ablation.sh to generate the metrics used in this script.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})

# Plot OOD train size vs metric plot
fn_template = "{id_dataset}_s-0_augment-True_oodsize-{data_size}_avg-False_dr-0.8_mlpsize-250_trn_prot-True.csv"

data_sizes = [10, 30, 100, 300, 1000, 3000, 10000, 30000]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=300)

near_fpr95_cifar10, near_auroc_cifar10 = [], []
far_fpr95_cifar10, far_auroc_cifar10 = [], []
near_fpr95_cifar100, near_auroc_cifar100 = [], []
far_fpr95_cifar100, far_auroc_cifar100 = [], []

# Gather metrics
id_dataset = "cifar10"
for data_size in data_sizes:
    csv_path = "metrics/" + fn_template.format(id_dataset=id_dataset, data_size=data_size)
    metrics = pd.read_csv(csv_path, index_col=0)
    near_fpr95_cifar10.append(metrics.loc[metrics.index=="nearood", "FPR@95"].iloc[0])
    near_auroc_cifar10.append(metrics.loc[metrics.index=="nearood", "AUROC"].iloc[0])
    far_fpr95_cifar10.append(metrics.loc[metrics.index=="farood", "FPR@95"].iloc[0])
    far_auroc_cifar10.append(metrics.loc[metrics.index=="farood", "AUROC"].iloc[0])

id_dataset = "cifar100"
for data_size in data_sizes:
    csv_path = "metrics/" + fn_template.format(id_dataset=id_dataset, data_size=data_size)
    metrics = pd.read_csv(csv_path, index_col=0)
    near_fpr95_cifar100.append(metrics.loc[metrics.index=="nearood", "FPR@95"].iloc[0])
    near_auroc_cifar100.append(metrics.loc[metrics.index=="nearood", "AUROC"].iloc[0])
    far_fpr95_cifar100.append(metrics.loc[metrics.index=="farood", "FPR@95"].iloc[0])
    far_auroc_cifar100.append(metrics.loc[metrics.index=="farood", "AUROC"].iloc[0])


# AUROC
ax1.set_title("AUROC - OOD Training Dataset Size", fontsize=12)
ax1.plot(near_auroc_cifar10, linewidth=2, marker='o', markersize=5)
ax1.plot(far_auroc_cifar10, linewidth=2, marker='o', markersize=5)
ax1.plot(near_auroc_cifar100, linewidth=2, marker='o', markersize=5)
ax1.plot(far_auroc_cifar100, linewidth=2, marker='o', markersize=5)

ax1.set_xlabel("Number of Examples", fontsize=10)
ax1.set_ylabel("AUROC", fontsize=10)
ax1.grid(True, linestyle='--', linewidth=0.5)
ax1.set_xticks(np.arange(len(data_sizes)), data_sizes, rotation=0, fontsize=12)
ax1.legend(["Near-OOD, ID=CIFAR10 ", "Far-OOD, ID=CIFAR10", "Near-OOD, ID=CIFAR100 ", "Far-OOD, ID=CIFAR100"], fontsize=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# FPR95
ax2.set_title("FPR95 - OOD Training Dataset Size", fontsize=12)
ax2.plot(near_fpr95_cifar10, linewidth=2, marker='o', markersize=5)
ax2.plot(far_fpr95_cifar10, linewidth=2, marker='o', markersize=5)
ax2.plot(near_fpr95_cifar100, linewidth=2, marker='o', markersize=5)
ax2.plot(far_fpr95_cifar100, linewidth=2, marker='o', markersize=5)

ax2.set_xlabel("Number of Examples", fontsize=10)
ax2.set_ylabel("FPR95", fontsize=10)
ax2.grid(True, linestyle='--', linewidth=0.5)
ax2.set_xticks(np.arange(len(data_sizes)), data_sizes, rotation=0, fontsize=12)
ax2.legend(["Near-OOD, ID=CIFAR10 ", "Far-OOD, ID=CIFAR10", "Near-OOD, ID=CIFAR100 ", "Far-OOD, ID=CIFAR100"], fontsize=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Save the figure
plt.tight_layout()
fig.savefig('plots/metric_per_ood_train_size.png', dpi=300, bbox_inches='tight')
plt.show()


# %%
# Plot metrics for each layer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})

id_dataset = "cifar100"
layer_names = []
layer_i_to_fn = []
layer_i_to_fn.append(f"{id_dataset}_s-0_augment-True_oodsize-all_avg-False_dr-0.8_mlpsize-250_trn_prot-True_protlayer-conv1_protch-64_nprot-100.csv")
layer_i_to_fn.append(f"{id_dataset}_s-0_augment-True_oodsize-all_avg-False_dr-0.8_mlpsize-250_trn_prot-True_protlayer-bn1_protch-64_nprot-100.csv")
layer_names.extend(["Conv1", "Bn1"])

for i_block in range(4):
    n_ch = 64 * 2**(i_block)
    for i_part in range(2):
        layer_i_to_fn.append(f"{id_dataset}_s-0_augment-True_oodsize-all_avg-False_dr-0.8_mlpsize-250_trn_prot-True_protlayer-{i_block}_{i_part}_conv1_protch-{n_ch}_nprot-100.csv")
        layer_i_to_fn.append(f"{id_dataset}_s-0_augment-True_oodsize-all_avg-False_dr-0.8_mlpsize-250_trn_prot-True_protlayer-{i_block}_{i_part}_bn1_protch-{n_ch}_nprot-100.csv")
        layer_i_to_fn.append(f"{id_dataset}_s-0_augment-True_oodsize-all_avg-False_dr-0.8_mlpsize-250_trn_prot-True_protlayer-{i_block}_{i_part}_conv2_protch-{n_ch}_nprot-100.csv")
        layer_i_to_fn.append(f"{id_dataset}_s-0_augment-True_oodsize-all_avg-False_dr-0.8_mlpsize-250_trn_prot-True_protlayer-{i_block}_{i_part}_bn2_protch-{n_ch}_nprot-100.csv")
        layer_names.append(f"Block{i_block} Part{i_part} Conv1")
        layer_names.append(f"Block{i_block} Part{i_part} BN1")
        layer_names.append(f"Block{i_block} Part{i_part} Conv2")
        layer_names.append(f"Block{i_block} Part{i_part} BN2")

layer_i_to_fn.append(f"{id_dataset}_s-0_augment-True_oodsize-all_avg-False_dr-0.8_mlpsize-250_trn_prot-True_protlayer-penultimate_protch-512_nprot-100.csv")
layer_i_to_fn.append(f"{id_dataset}_s-0_augment-True_oodsize-all_avg-False_dr-0.8_mlpsize-250_trn_prot-True_protlayer-fc_protch-100_nprot-100.csv")
layer_names.append("Penultimate")  
layer_names.append("FC - Logit Layer")

# Gather metrics
near_fpr95, near_auroc = [], []
far_fpr95, far_auroc = [], []
for fn in layer_i_to_fn:
    metrics = pd.read_csv(f"metrics/{fn}", index_col=0)
    near_fpr95.append(metrics.loc[metrics.index=="nearood", "FPR@95"].iloc[0])
    near_auroc.append(metrics.loc[metrics.index=="nearood", "AUROC"].iloc[0])
    far_fpr95.append(metrics.loc[metrics.index=="farood", "FPR@95"].iloc[0])
    far_auroc.append(metrics.loc[metrics.index=="farood", "AUROC"].iloc[0])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), dpi=300)
ax1.set_title("AUROC - OOD Classification Using a Single Layer - ID Dataset: CIFAR100", fontsize=12)
ax1.plot(near_auroc, linewidth=2, marker='o', markersize=5)
ax1.plot(far_auroc, linewidth=2, marker='o', markersize=5)
ax1.set_xlabel("Layer Name", fontsize=10)
ax1.set_ylabel("AUROC", fontsize=10)
ax1.grid(True, linestyle='--', linewidth=0.5)
ax1.set_xticks(np.arange(len(layer_names)), layer_names, rotation=75, fontsize=12)
ax1.legend(["Near-OOD Datasets", "Far-OOD Datasets"], fontsize=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

multilayer_auroc = 79.38
ax1.axhline(y=multilayer_auroc, color='b', linestyle='--', linewidth=1)
ax1.annotate(f'Multiple Layers Near-OOD AUROC: {multilayer_auroc}', 
             xy=(0, multilayer_auroc),
             xytext=(0.5, multilayer_auroc + 0.5),
             arrowprops=dict(facecolor='blue', shrink=0.05),
             fontsize=10, color='black')

multilayer_auroc = 89.81
ax1.axhline(y=multilayer_auroc, color='orange', linestyle='--', linewidth=1)
ax1.annotate(f'Multiple Layers Far-OOD AUROC: {multilayer_auroc}', 
             xy=(0, multilayer_auroc), 
             xytext=(0.5, multilayer_auroc + 0.5),
             arrowprops=dict(facecolor='orange', shrink=0.05),
             fontsize=10, color='black')

ax2.set_title("FPR95 - OOD Classification Using a Single Layer - ID Dataset: CIFAR100", fontsize=12)
ax2.plot(near_fpr95, linewidth=2, marker='o', markersize=5)
ax2.plot(far_fpr95, linewidth=2, marker='o', markersize=5)
ax2.set_xlabel("Layer Name", fontsize=10)
ax2.set_ylabel("FPR95", fontsize=10)
ax2.grid(True, linestyle='--', linewidth=0.5)
ax2.set_xticks(np.arange(len(layer_names)), layer_names, rotation=75, fontsize=12)
ax2.legend(["Near-OOD Datasets", "Far-OOD Datasets"], fontsize=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

multilayer_fpr = 59.61
ax2.axhline(y=multilayer_fpr, color='b', linestyle='--', linewidth=1)
ax2.annotate(f'Multiple Layers Near-OOD FPR95: {multilayer_fpr}', 
             xy=(0, multilayer_fpr),
             xytext=(0.5, multilayer_fpr + 0.5),
             arrowprops=dict(facecolor='blue', shrink=0.05),
             fontsize=10, color='black')

multilayer_fpr = 40.37
ax2.axhline(y=multilayer_fpr, color='orange', linestyle='--', linewidth=1)
ax2.annotate(f'Multiple Layers Far-OOD FPR95: {multilayer_fpr}', 
             xy=(0, multilayer_fpr), 
             xytext=(0.5, multilayer_fpr + 0.5),
             arrowprops=dict(facecolor='orange', shrink=0.05),
             fontsize=10, color='black')


plt.tight_layout()

# Save the figure
fig.savefig('plots/metric_per_layer.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
