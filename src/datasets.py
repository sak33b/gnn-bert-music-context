"""
datasets.py — NOT explicitly named in the assignment's folder diagram, but
train.py needs somewhere to turn (processed graphs + metadata CSVs) into
PyTorch batches, so that glue code lives here rather than duplicated across
train.py's four task branches.

Reads:
  data/processed/graphs/{track_id}.pt        (from graph_builder.py)
  data/raw/fma_metadata/tracks.csv            (genre/tag labels)
  data/raw/magnatagatune/annotations_final.csv
  data/raw/musiccaps/musiccaps-public.csv     (captions, for Task 1/3/4 text)
  data/raw/deam/annotations/*.csv             (valence/arousal, for Task 3 aux loss)
  data/splits/{dataset}_{split}.json          (track_id lists, no artist leakage)
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from bert_encoder import get_tokenizer, tokenize_batch


def load_split(splits_dir: str, dataset_name: str, split: str) -> list:
    path = os.path.join(splits_dir, f"{dataset_name}_{split}.json")
    with open(path, "r") as f:
        return json.load(f)


def load_tag_table(raw_dir: str, dataset_name: str, num_tags: int) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by track_id with `num_tags` binary tag columns
    (the top-K most frequent tags in the dataset, per spec 4.1's "top-50 tags").
    """
    if dataset_name == "musiccaps":
        # MusicCaps has captions and AudioSet labels, but no ready-made music
        # tag matrix.  The annotated aspect list is a strong, transparent tag
        # proxy: frequent aspects (for example guitar, piano, singing) become
        # the multi-label targets used by Tasks 1-3.
        import ast
        path = os.path.join(raw_dir, "musiccaps", "musiccaps-public.csv")
        df = pd.read_csv(path)
        aspects = df["aspect_list"].fillna("[]").map(ast.literal_eval)
        counts = {}
        for values in aspects:
            for tag in values:
                clean = str(tag).strip().lower()
                if clean:
                    counts[clean] = counts.get(clean, 0) + 1
        top_tags = [tag for tag, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:num_tags]]
        matrix = np.zeros((len(df), len(top_tags)), dtype=np.float32)
        tag_index = {tag: i for i, tag in enumerate(top_tags)}
        for row, values in enumerate(aspects):
            for tag in values:
                index = tag_index.get(str(tag).strip().lower())
                if index is not None:
                    matrix[row, index] = 1.0
        return pd.DataFrame(matrix, index=df["ytid"].astype(str), columns=top_tags)
    if dataset_name == "magnatagatune":
        df = pd.read_csv(os.path.join(raw_dir, "magnatagatune", "annotations_final.csv"), sep="\t")
        df = df.set_index("clip_id")
        tag_cols = [c for c in df.columns if c not in ("mp3_path",)]
        top_tags = df[tag_cols].sum().sort_values(ascending=False).index[:num_tags]
        return df[top_tags]
    elif dataset_name == "fma_medium":
        df = pd.read_csv(os.path.join(raw_dir, "fma_metadata", "tracks.csv"), index_col=0, header=[0, 1])
        genres = pd.get_dummies(df[("track", "genre_top")])
        top = genres.sum().sort_values(ascending=False).index[:num_tags]
        return genres[top]
    else:
        raise ValueError(f"Unknown tag dataset: {dataset_name}")


def load_captions(raw_dir: str) -> dict:
    """track_id -> caption string, from MusicCaps' public CSV (column names per Google's release)."""
    path = os.path.join(raw_dir, "musiccaps", "musiccaps-public.csv")
    df = pd.read_csv(path)
    return dict(zip(df["ytid"], df["caption"]))


def load_emotion_labels(raw_dir: str) -> dict:
    """track_id -> (mean_valence, mean_arousal), scaled from DEAM's [1,9] to [-1,1]."""
    anno_dir = os.path.join(raw_dir, "deam", "annotations")
    static_path = os.path.join(anno_dir, "static_annotations.csv")
    if not os.path.exists(static_path):
        return {}
    df = pd.read_csv(static_path)
    df.columns = [c.strip() for c in df.columns]
    out = {}
    for _, row in df.iterrows():
        v = (row["valence_mean"] - 5.0) / 4.0
        a = (row["arousal_mean"] - 5.0) / 4.0
        out[str(int(row["song_id"]))] = (v, a)
    return out


class MusicContextDataset(Dataset):
    """
    One example = one track: its structure graph, tag vector, optional caption
    text, and optional emotion targets. `task` controls which fields are
    actually populated (Task 1 doesn't need graphs; Task 2 doesn't need text).
    """

    def __init__(self, track_ids: list, cfg: dict, tag_table: pd.DataFrame,
                 captions: dict = None, emotions: dict = None, task: str = "fusion"):
        graph_dir = os.path.join(cfg["paths"]["processed_dir"], "graphs")
        self.track_ids = [str(t) for t in track_ids if str(t) in tag_table.index
                          and (task == "bert" or os.path.exists(os.path.join(graph_dir, f"{t}.pt")))]
        self.cfg = cfg
        self.tag_table = tag_table
        self.captions = captions or {}
        self.emotions = emotions or {}
        self.task = task
        self.graph_dir = graph_dir
        self.tokenizer = get_tokenizer(cfg["text"]["bert_model_name"]) if task in ("bert", "fusion", "contrastive") else None

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        tags = torch.tensor(self.tag_table.loc[track_id].values.astype(np.float32))

        item = {"track_id": track_id, "tags": tags}

        if self.task in ("gnn", "fusion", "contrastive"):
            graph_path = os.path.join(self.graph_dir, f"{track_id}.pt")
            item["graph"] = torch.load(graph_path, weights_only=False)

        if self.task in ("bert", "fusion", "contrastive"):
            caption = self.captions.get(track_id, "")
            item["caption"] = caption

        if self.task == "fusion":
            v, a = self.emotions.get(track_id, (0.0, 0.0))
            item["emotion"] = torch.tensor([v, a], dtype=torch.float32)
            item["has_emotion"] = track_id in self.emotions

        return item


def collate_fn(batch: list, tokenizer, max_length: int, task: str):
    """
    Custom collate: PyG graphs need Batch.from_data_list (not the default
    torch stacking), and captions need on-the-fly BERT tokenization.
    """
    out = {"track_id": [b["track_id"] for b in batch], "tags": torch.stack([b["tags"] for b in batch])}

    if task in ("gnn", "fusion", "contrastive"):
        out["graph"] = Batch.from_data_list([b["graph"] for b in batch])

    if task in ("bert", "fusion", "contrastive"):
        texts = [b["caption"] for b in batch]
        tok = tokenize_batch(tokenizer, texts, max_length)
        out["input_ids"] = tok["input_ids"]
        out["attention_mask"] = tok["attention_mask"]

    if task == "fusion":
        out["emotion"] = torch.stack([b["emotion"] for b in batch])
        out["has_emotion"] = torch.tensor([b["has_emotion"] for b in batch], dtype=torch.bool)

    return out
