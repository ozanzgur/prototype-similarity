import torch
from torch import nn
import os.path as osp
from PIL import Image
import numpy as np

from openood.postprocessors import BasePostprocessor
import openood.utils.comm as comm

def subsample_dataset(dataset, sample_size):
    idx = np.random.permutation(len(dataset))
    idx = [int(i) for i in idx[:sample_size]]
    return torch.utils.data.Subset(dataset, indices=idx)

class ToRGB:
    def __call__(self, img):
        return img.convert('RGB')

class PrototypePostprocessor(BasePostprocessor):
    def __init__(self, config, scaler, ood_classifer):
        super(PrototypePostprocessor, self).__init__(config)
        self.scaler = scaler
        self.ood_classifer = ood_classifer
        self.APS_mode = False # hparam search

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, reconstruction_model, data):
        features = reconstruction_model.prototype_scores(data.cuda())
        features = self.scaler.transform(features)
        features = torch.tensor(features).cuda()
        y_pred = self.ood_classifer(features).cpu()
        pred = y_pred[:, 1] > 0.5
        conf = y_pred[:, 1]
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
                    #if "/tin/test" in filename:
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