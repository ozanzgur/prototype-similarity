import torch
from torch import nn
import os.path as osp
from PIL import Image

from openood.postprocessors import BasePostprocessor
import openood.utils.comm as comm

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
    def __init__(self, txt_file, img_dir, transform=None):
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
                    #if "/tin/test" in filename:
                    self.image_labels.append((filename, int(label)))
                else:
                    raise ValueError(f"Line in text file is not in expected format: {line}")

        self.transform = transform

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name, label = self.image_labels[idx]
        img_name = osp.join(self.img_dir, img_name)
        image = Image.open(img_name)  # Convert image to RGB

        if self.transform:
            image = self.transform(image)

        return image, label