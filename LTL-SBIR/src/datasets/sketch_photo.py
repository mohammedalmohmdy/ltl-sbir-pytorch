
import os
from typing import Optional
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SketchPhotoPairDataset(Dataset):
    """Loads sketches and photos with labels.
    If `train_pairs_csv` provided, yields explicit triplets (S, I_pos, I_neg).
    Else yields mixed (S, I) to support batch‑hard triplet mining.
    """
    def __init__(self, root: str, sketches_csv: str, images_csv: str,
                 train_pairs_csv: Optional[str] = None, img_size: int = 224, split: str = "train"):
        self.root = root
        self.sketches = pd.read_csv(os.path.join(root, sketches_csv))
        self.images = pd.read_csv(os.path.join(root, images_csv))
        self.sketches['abs_path'] = self.sketches['path'].apply(lambda p: os.path.join(root, p))
        self.images['abs_path'] = self.images['path'].apply(lambda p: os.path.join(root, p))
        self.split = split

        if train_pairs_csv is not None and os.path.exists(os.path.join(root, train_pairs_csv)):
            self.train_pairs_csv = os.path.join(root, train_pairs_csv)
            self.pairs = pd.read_csv(self.train_pairs_csv)
            self.pairs['sketch_abs'] = self.pairs['sketch_path'].apply(lambda p: os.path.join(root, p))
            self.pairs['pos_abs'] = self.pairs['pos_img_path'].apply(lambda p: os.path.join(root, p))
            self.pairs['neg_abs'] = self.pairs['neg_img_path'].apply(lambda p: os.path.join(root, p))
        else:
            self.train_pairs_csv = None

        norm_mean = [0.485, 0.456, 0.406]
        norm_std  = [0.229, 0.224, 0.225]
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

        self.image_labels = dict(zip(self.images['abs_path'].tolist(), self.images['label'].tolist()))

    def __len__(self):
        return len(self.pairs) if self.train_pairs_csv else len(self.sketches)

    def _load(self, path: str):
        return self.tf(Image.open(path).convert("RGB"))

    def __getitem__(self, idx: int):
        if self.train_pairs_csv:
            row = self.pairs.iloc[idx]
            s = self._load(row['sketch_abs'])
            ip = self._load(row['pos_abs'])
            ineg = self._load(row['neg_abs'])
            y = int(self.image_labels[row['pos_abs']])
            y_neg = int(self.image_labels[row['neg_abs']])
            return {"sketch": s, "img_pos": ip, "img_neg": ineg, "y": y, "y_neg": y_neg}
        else:
            srow = self.sketches.iloc[idx]
            s = self._load(srow['abs_path'])
            # sample any image; labels used for batch‑hard mining when concatenated
            irow = self.images.sample(1).iloc[0]
            i = self._load(irow['abs_path'])
            ys = int(srow['label']); yi = int(irow['label'])
            return {"sketch": s, "image": i, "y_sketch": ys, "y_image": yi}

def build_eval_sets(root: str, sketches_csv: str, images_csv: str, img_size: int = 224):
    ds = SketchPhotoPairDataset(root, sketches_csv, images_csv, None, img_size, split="eval")
    sketches = ds.sketches[['abs_path','label']].values.tolist()
    images = ds.images[['abs_path','label']].values.tolist()
    tf = ds.tf
    return sketches, images, tf
