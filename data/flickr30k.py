import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class Flickr30kDataset(Dataset):
    def __init__(self, root, split='train', image_size=224):
        self.root = root
        self.split = split
        ann = os.path.join(root, 'annotations', f'{split}.json')
        with open(ann, 'r', encoding='utf-8') as f:
            self.items = json.load(f)
        self.tfm = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = Image.open(os.path.join(self.root, 'images', item['image'])).convert('RGB')
        img = self.tfm(img)
        txt = item['caption']
        return img, txt