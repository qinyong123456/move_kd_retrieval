import os
import json
from PIL import Image
import re
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class Flickr30kDataset(Dataset):
    def __init__(self, root=None, split='train', image_size=224, images_dir=None, captions_txt=None, val_ratio=0.05):
        self.root = root
        self.split = split
        self.images_dir = images_dir
        self.captions_txt = captions_txt
        if captions_txt and images_dir:
            with open(captions_txt, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip()]
            pairs = []
            pat = re.compile(r"^\s*([^\s,#]+\.(?:jpg|jpeg|png))[,#\s]*(?:\d+)?\s*(.*)$", re.IGNORECASE)
            for l in lines:
                m = pat.match(l)
                if not m:
                    continue
                img_name = m.group(1).strip()
                cap = m.group(2).strip().strip('"')
                pairs.append({'image': img_name, 'caption': cap})
            order = []
            seen = set()
            for p in pairs:
                if p['image'] not in seen:
                    seen.add(p['image'])
                    order.append(p['image'])
            n = len(order)
            vn = max(1, int(n * val_ratio))
            val_set = set(order[-vn:])
            if split == 'train':
                self.items = [p for p in pairs if p['image'] not in val_set]
            else:
                kept = {}
                for p in pairs:
                    if p['image'] in val_set and p['image'] not in kept:
                        kept[p['image']] = p
                self.items = list(kept.values())
        else:
            ann = os.path.join(root, 'annotations', f'{split}.json')
            with open(ann, 'r', encoding='utf-8') as f:
                self.items = json.load(f)
            self.images_dir = os.path.join(root, 'images')
        self.tfm = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = Image.open(os.path.join(self.images_dir, item['image'])).convert('RGB')
        img = self.tfm(img)
        txt = item['caption']
        return img, txt