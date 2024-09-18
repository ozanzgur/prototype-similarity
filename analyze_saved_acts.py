# Observations:
# In OOD data, cifar10 acts show a lower min similarity to the corner embedding
# CIFAR10 ranges:
#     OOD: -0.5, 0.3, ID: 0.5, 0.675
# IMAGENET ranges: (layer 4)
#     OOD: 0.76, 0.86, ID: 0.76, 0.9
# IMAGENET ranges: (layer 3)
#     OOD: 0.2, 0.45, ID: 0.26, 0.36

# INET has a much lower difference in min similarities to the corner (layer 3)

# In Imagenet activations, ood centers are more similar to id centers
# But, Imagenet has more variety among the ID activations from different locations

# In imagenet, similarity to center vector is larger in OOD
# In cifar, similarity to center vector is lower in OOD
# In imagenet, OOD activations gather around a similar point in OOD examples
# In cifar, they scatter more
# Hypothesis: in imagenet, fewer dimensions show variations in OOD examples


# EXPERIMENT:
# Try other features like min similarity
# Try with spatial avg of similarities
# Add a dim reduction layer before prototypes in scoring stage or mask features
#     Add sigmoid weights to each dimension before distance calculation in ood detection stage

# %%
import torch
import seaborn as sns
import matplotlib.pyplot as plt

def load_cifar_acts():
    ood_acts = torch.load('saved_acts/cifar10/saved_acts-prot_scores-eval-2000.pt') # Lower min similarity to corner
    det_train_acts = torch.load('saved_acts/cifar10/saved_acts-prot_scores-train_detector-0.pt')
    prot_train_acts = torch.load('saved_acts/cifar10/saved_acts-train_orig_acts-train_prototypes-0.pt') # Higher min similarity to corner
    return ood_acts, det_train_acts, prot_train_acts

def load_imagenet_acts():
    ood_acts = torch.load('saved_acts/imagenet200/saved_acts-prot_scores-eval-2000.pt')
    det_train_acts = torch.load('saved_acts/imagenet200/saved_acts-prot_scores-train_detector-0.pt')
    prot_train_acts = torch.load('saved_acts/imagenet200/saved_acts-train_orig_acts-train_prototypes-0.pt') # Higher min similarity to corner
    return ood_acts, det_train_acts, prot_train_acts

cifar10_ood_acts, _, cifar10_id_acts = load_cifar_acts()
imagenet_ood_acts, _, imagenet_id_acts = load_imagenet_acts()

# %%

# Measure min similarity to the id center vector
cifar_mins = []
inet_mins = []

for i_ex in range(10):
    h2 = imagenet_ood_acts[0][i_ex].shape[-1]
    #v2_id = imagenet_id_acts[0][i_ex, :, h2//2, h2//2]
    v2_id = imagenet_id_acts[0][i_ex, :, 0, 0]
    v2_id = v2_id / torch.norm(v2_id, 2)


    a2 = imagenet_ood_acts[0][i_ex].flatten(start_dim=1)
    a2_norm = a2 / torch.norm(a2, dim=0, keepdim=True, p=2)
    a2_cos = (a2_norm * v2_id.unsqueeze(-1)).sum(dim=0)
    a2_cos = a2_cos.reshape((h2, h2))
    #imagenet_min = a2_cos.cpu().numpy().min()
    imagenet_min = a2_cos.cpu().numpy()[h2//2, h2//2]
    inet_mins.append(imagenet_min)
    #plt.figure()
    #sns.heatmap(a2_cos.cpu().numpy()

    h1 = cifar10_ood_acts[0][i_ex].shape[-1]
    #v1_id = cifar10_id_acts[0][i_ex, :, h1//2, h1//2]
    v1_id = cifar10_id_acts[0][i_ex, :, 0, 0]
    v1_id = v1_id / torch.norm(v1_id, 2)

    a1 = cifar10_ood_acts[0][i_ex].flatten(start_dim=1)
    a1_norm = a1 / torch.norm(a1, dim=0, keepdim=True, p=2)

    a1_cos = (a1_norm * v1_id.unsqueeze(-1)).sum(dim=0)
    a1_cos = a1_cos.reshape((h1, h1))
    #cifar_min = a1_cos.cpu().numpy().min()
    cifar_min = a1_cos.cpu().numpy()[h1//2, h1//2]
    cifar_mins.append(cifar_min)
    #plt.figure()
    #sns.heatmap(a1_cos.cpu().numpy())


plt.figure()
sns.histplot(cifar_mins)
sns.histplot(inet_mins)

# %%

# Measure min similarity to the corner vector
cifar_mins = []
inet_mins = []

for i_ex in range(10):
    # inet_acts = imagenet_ood_acts
    # cifar_acts = cifar10_ood_acts
    inet_acts = imagenet_id_acts
    cifar_acts = cifar10_id_acts

    h2 = inet_acts[0][i_ex].shape[-1]
    a2 = inet_acts[0][i_ex].flatten(start_dim=1)
    v2 = a2[:, (h2//2)**2]
    v2 = v2 / torch.norm(v2, 2)
    a2_norm = a2 / torch.norm(a2, dim=0, keepdim=True, p=2)
    a2_cos = (a2_norm * v2.unsqueeze(-1)).sum(dim=0)
    a2_cos = a2_cos.reshape((h2, h2))
    imagenet_min = a2_cos.cpu().numpy().min()
    inet_mins.append(imagenet_min)
    #plt.figure()
    #sns.heatmap(a2_cos.cpu().numpy()

    h1 = cifar_acts[0][i_ex].shape[-1]
    a1 = cifar_acts[0][i_ex].flatten(start_dim=1)
    v1 = a1[:, (h1//2)**2]
    v1 = v1 / torch.norm(v1, 2)
    a1_norm = a1 / torch.norm(a1, dim=0, keepdim=True, p=2)
    a1_cos = (a1_norm * v1.unsqueeze(-1)).sum(dim=0)
    a1_cos = a1_cos.reshape((h1, h1))
    cifar_min = a1_cos.cpu().numpy().min()
    cifar_mins.append(cifar_min)
    #plt.figure()
    #sns.heatmap(a1_cos.cpu().numpy())

plt.figure()
sns.histplot(cifar_mins)
sns.histplot(inet_mins)

# %%


# %%

i_ex = 3

h1 = cifar10_ood_acts[0][i_ex].shape[-1]
a1 = cifar10_ood_acts[0][i_ex].flatten(start_dim=1)
v1 = a1[:, 0]
v1 = v1 / torch.norm(v1, 2)
a1_norm = a1 / torch.norm(a1, dim=0, keepdim=True, p=2)
a1_cos = (a1_norm * v1.unsqueeze(-1)).sum(dim=0)
a1_cos = a1_cos.reshape((h1, h1))
#plt.figure()
#sns.heatmap(a1_cos.cpu().numpy())

h2 = imagenet_ood_acts[0][i_ex].shape[-1]
a2 = imagenet_ood_acts[0][i_ex].flatten(start_dim=1)
v2 = a2[:, (h2 // 2)**2]
v2 = v2 / torch.norm(v2, 2)
a2_norm = a2 / torch.norm(a2, dim=0, keepdim=True, p=2)
a2_cos = (a2_norm * v2.unsqueeze(-1)).sum(dim=0)
a2_cos = a2_cos.reshape((h2, h2))
plt.figure()
sns.heatmap(a2_cos.cpu().numpy())



# %%