import torch
from torch import nn
import os.path as osp
from PIL import Image
from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})

from openood.postprocessors import BasePostprocessor
import openood.utils.comm as comm

project_dir = str(Path(__file__).resolve().parents[1])

def subsample_dataset(dataset, sample_size):
    idx = np.random.permutation(len(dataset))
    idx = [int(i) for i in idx[:sample_size]]
    return torch.utils.data.Subset(dataset, indices=idx)

class ToRGB:
    def __call__(self, img):
        return img.convert('RGB')

class PrototypePostprocessor(BasePostprocessor):
    def __init__(self, config, ood_classifier):
        super(PrototypePostprocessor, self).__init__(config)
        #self.scaler = scaler
        self.ood_classifier = ood_classifier
        self.APS_mode = False # hparam search

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, reconstruction_model, data):
        y_pred = self.ood_classifier(data.cuda()).cpu()
        pred = y_pred > 0.5
        conf = y_pred
        return pred, conf


class ImageDatasetFromFile(torch.utils.data.Dataset):
    def __init__(self, txt_file, img_dir, transform=None, is_imagenet=False):
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
                    if is_imagenet:
                        filename = self.imgnet_format_convert(filename)
                    
                    if osp.exists(osp.join(self.img_dir, filename)):
                        self.image_labels.append((filename, int(label)))
                else:
                    raise ValueError(f"Line in text file is not in expected format: {line}")

        print(f"Existing images in file {txt_file}: {len(self.image_labels)}")
        self.transform = transform

    def __len__(self):
        return len(self.image_labels)
    
    def imgnet_format_convert(self, filename):
        #imagenet_1k/train/n01443537/n01443537_10007.JPEG 0
        dataset_name, set_name, dirname, filename = filename.split('/')
        img_name, ext = filename.split('.')
        return f"{dataset_name}/{set_name}/{img_name}_{dirname}.{ext}"

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name, label = self.image_labels[idx]
        img_name = osp.join(self.img_dir, img_name)
        image = Image.open(img_name)  # Convert image to RGB

        if self.transform:
            image = self.transform(image)

        return image, label
    
class MergedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.length1 = len(self.dataset1)
        self.length2 = len(self.dataset2)
        self.total_length = self.length1 + self.length2
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, index):
        if index < self.length1:
            # Fetch from the first dataset and assign label 1
            data = self.dataset1[index][0]
            label = torch.tensor([0], dtype=torch.float32)

        else:
            # Fetch from the second dataset and assign label 0
            data = self.dataset2[index - self.length1][0]
            label = torch.tensor([1], dtype=torch.float32)

        return data, label
    
def plot_prot_usage(model):
    print("Saving prototype usage plots")
    import seaborn as sns
    prot_usage = model.prototype_matchers[0].prototype_usage_counts.cpu().detach().numpy()
    total_usage_rates = prot_usage.sum(axis=1)
    usage_order_idx = np.argsort(total_usage_rates)

    plot_dir = osp.join(project_dir, "plots")
    if not osp.exists(plot_dir):
        os.mkdir(plot_dir)

    for prot_order in range(20):
        plt.figure(figsize=(6, 6), dpi=300)
        usage_rate = prot_usage[usage_order_idx[-prot_order-1]]
        h = int(np.sqrt(usage_rate.shape))
        heatmap = sns.heatmap(usage_rate.reshape(h, h), cmap='viridis', xticklabels=False, yticklabels=False)
        plt.title(f"Most Used Prototype  {prot_order}\nMatch Count in Spatial Dimensions")

        # Adjust the color bar
        cbar = heatmap.collections[0].colorbar
        cbar.set_label('Selection Count for Reconstruction During Training', fontsize=10)
        # Remove axis ticks
        heatmap.tick_params(left=False, bottom=False)

        plt.savefig(osp.join(plot_dir, f'prot_usage_top_{prot_order}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
