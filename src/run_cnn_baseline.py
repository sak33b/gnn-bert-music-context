"""Train and evaluate the Task 2 mel-spectrogram CNN baseline.

This baseline deliberately sees neither captions nor graph edges.  It is kept
in a separate small script so a fair audio-only reference can be reproduced
without changing the main four-task CLI.
"""
import json
import os

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from datasets import load_split, load_tag_table
from evaluate import tag_metrics
from gnn_model import CNNBaseline
from utils import get_device, load_config, set_seed


class MelDataset(Dataset):
    def __init__(self, ids, tags, feature_dir):
        self.ids = [str(i) for i in ids if str(i) in tags.index and os.path.exists(os.path.join(feature_dir, f"{i}.npz"))]
        self.tags, self.feature_dir = tags, feature_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        track_id = self.ids[index]
        mel = np.load(os.path.join(self.feature_dir, f"{track_id}.npz"))["mel"]
        # Join fixed segments into one time axis and use a fixed-width crop.
        mel = mel.reshape(-1, mel.shape[-1])[:256]
        if len(mel) < 256:
            mel = np.pad(mel, ((0, 256 - len(mel)), (0, 0)))
        return torch.tensor(mel.T[None], dtype=torch.float32), torch.tensor(self.tags.loc[track_id].values, dtype=torch.float32)


def main():
    cfg = load_config(); set_seed(cfg["train"]["seed"])
    device = get_device(cfg["train"]["device"])
    tags = load_tag_table(cfg["paths"]["raw_dir"], "musiccaps", cfg["model"]["num_tags"])
    feature_dir = os.path.join(cfg["paths"]["processed_dir"], "features")
    splits = {name: MelDataset(load_split(cfg["paths"]["splits_dir"], "musiccaps", name), tags, feature_dir) for name in ("train", "test")}
    loaders = {name: DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=name == "train") for name, ds in splits.items()}
    model = CNNBaseline(cfg["audio"]["n_mels"], cfg["model"]["num_tags"], cfg["model"]["dropout"]).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"])
    model.train()
    for mel, target in loaders["train"]:
        logits = model(mel.to(device)); loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target.to(device))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval(); probs, truth = [], []
    with torch.no_grad():
        for mel, target in loaders["test"]:
            probs.append(torch.sigmoid(model(mel.to(device))).cpu().numpy()); truth.append(target.numpy())
    metrics = tag_metrics(np.concatenate(truth), np.concatenate(probs))
    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg["paths"]["checkpoint_dir"], "cnn_baseline.pt"))
    path = os.path.join(cfg["paths"]["results_dir"], "metrics.json")
    with open(path) as handle: all_metrics = json.load(handle)
    all_metrics["task2_cnn_baseline"] = metrics
    with open(path, "w") as handle: json.dump(all_metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
