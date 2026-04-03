from pathlib import Path
import torch
from torch.utils.data import Dataset
from dataset_utils import pad_video


class AdsDataset(Dataset):

    def __init__(self, dataset_dir, max_shots=40):
        self.files = sorted(Path(dataset_dir).glob("*.pt"))
        self.max_shots = max_shots

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        item = torch.load(self.files[idx], map_location="cpu")

        E = item["E"]
        mask = item["mask"]
        dur = item["dur"]

        # pad to fixed length
        E, mask, dur, valid = pad_video(E, mask, dur, self.max_shots)

        return E, mask, dur, valid
    
dataset = AdsDataset("..\dataset")
E, mask, dur, valid = dataset[0]
print("E shape:", E.shape)
print("mask shape:", mask.shape)
print("dur shape:", dur.shape)