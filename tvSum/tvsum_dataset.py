import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class TVSumShotDataset(Dataset):
    def __init__(self, json_dir, embedding_dir):
        self.json_dir = json_dir
        self.embedding_dir = embedding_dir

        self.samples = []

        for file in os.listdir(json_dir):
            if not file.endswith(".json"):
                continue

            video_name = os.path.splitext(file)[0]
            json_path = os.path.join(json_dir, file)
            emb_path = os.path.join(embedding_dir, f"{video_name}.npy")

            if not os.path.exists(emb_path):
                print(f"Missing embedding for: {video_name}")
                continue

            self.samples.append((json_path, emb_path, video_name))

        self.samples.sort(key=lambda x: x[2])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        json_path, emb_path, video_name = self.samples[idx]

        with open(json_path, "r", encoding="utf-8") as f:
            shots = json.load(f)

        E = np.load(emb_path)  # [N, D]

        scores = []
        durations = []

        for shot in shots:
            scores.append(float(shot["score"]))
            durations.append(float(shot["end"] - shot["start"]))

        E = torch.tensor(E, dtype=torch.float32)
        scores = torch.tensor(scores, dtype=torch.float32)
        durations = torch.tensor(durations, dtype=torch.float32)

        valid = torch.ones(len(scores), dtype=torch.float32)

        return E, scores, durations, valid, video_name