import os
import torch
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FoodDataset(Dataset):
    def __init__(self, file_paths, labels, class_to_idx, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[self.labels[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_data_split(file_path, base_dir):
    paths, labels = [], []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                paths.append(os.path.join(base_dir, 'images', line + '.jpg'))
                labels.append(line.split('/')[0])
    return paths, labels
