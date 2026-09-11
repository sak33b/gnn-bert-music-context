"""
train.py — single entrypoint for all four tasks. Select which to run with
--task {bert, gnn, fusion, contrastive}. Implements spec Algorithms 1-4
directly: each training loop below is a line-for-line match to the
pseudocode in the assignment (section 7).

Usage:
    python src/train.py --task bert
    python src/train.py --task gnn
    python src/train.py --task fusion --fusion_mode cross_attn
    python src/train.py --task contrastive
"""
import os
import json
import argparse
from functools import partial

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import average_precision_score, f1_score
import matplotlib.pyplot as plt

from utils import load_config, set_seed, get_device, count_parameters
from datasets import MusicContextDataset, load_split, load_tag_table, load_captions, load_emotion_labels, collate_fn
from bert_encoder import BertTagger, get_tokenizer, bce_multilabel_loss
from gnn_model import GNNTagger
from fusion_model import GNNBertFusion, multitask_loss
from contrastive import DualEncoder, info_nce_loss
from contrastive import retrieval_recall_at_k


def build_dataloaders(cfg: dict, task: str):
    raw_dir, splits_dir = cfg["paths"]["raw_dir"], cfg["paths"]["splits_dir"]
    audio_ds = cfg["datasets"]["active_audio"]

    tag_table = load_tag_table(raw_dir, audio_ds, cfg["model"]["num_tags"])
    captions = load_captions(raw_dir) if task in ("bert", "fusion", "contrastive") else None
    emotions = load_emotion_labels(raw_dir) if task == "fusion" else None

    loaders = {}
    for split in ("train", "val", "test"):
        track_ids = load_split(splits_dir, audio_ds, split)
        ds = MusicContextDataset(track_ids, cfg, tag_table, captions, emotions, task=task)
        tokenizer = ds.tokenizer
        loaders[split] = DataLoader(
            ds, batch_size=cfg["train"]["batch_size"], shuffle=(split == "train"),
            collate_fn=partial(collate_fn, tokenizer=tokenizer, max_length=cfg["text"]["max_length"], task=task),
            num_workers=2,
        )
    return loaders, tag_table.shape[1]


def to_device(batch: dict, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
        elif k == "graph":
            batch[k] = v.to(device)
    return batch


@torch.no_grad()
def validation_tag_metrics(model, loader, device, task: str) -> dict:
    """Compute held-out telemetry after each epoch without affecting gradients."""
    model.eval()
    probabilities, targets = [], []
    for batch in loader:
        batch = to_device(batch, device)
        if task == "bert":
            logits = model(batch["input_ids"], batch["attention_mask"])
        else:
            graph = batch["graph"]
            if task == "gnn":
                logits = model(graph.x, graph.edge_index, graph.batch)
            else:
                logits, _ = model(graph.x, graph.edge_index, graph.batch,
                                  batch["input_ids"], batch["attention_mask"])
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(batch["tags"].cpu().numpy())

    y_prob, y_true = np.concatenate(probabilities), np.concatenate(targets)
    y_pred = (y_prob >= 0.5).astype(int)
    valid = [index for index in range(y_true.shape[1]) if y_true[:, index].sum() > 0]
    auc_pr = float(np.mean([average_precision_score(y_true[:, index], y_prob[:, index])
                            for index in valid])) if valid else 0.0
    return {
        "val_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "val_micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "val_auc_pr": auc_pr,
    }


@torch.no_grad()
def validation_retrieval_metrics(model, loader, device, temperature: float) -> dict:
    """Compute validation InfoNCE and bidirectional Recall@K after each epoch."""
    model.eval()
    graph_embeddings, text_embeddings, losses = [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        graph = batch["graph"]
        graph_emb, text_emb = model(graph.x, graph.edge_index, graph.batch,
                                    batch["input_ids"], batch["attention_mask"])
        graph_embeddings.append(graph_emb.cpu())
        text_embeddings.append(text_emb.cpu())
        losses.append(info_nce_loss(graph_emb, text_emb, temperature).item())
    metrics = retrieval_recall_at_k(torch.cat(graph_embeddings), torch.cat(text_embeddings))
    metrics["val_info_nce"] = float(np.mean(losses))
    return {f"val_{key}": value for key, value in metrics.items()}


# --------------------------------------------------------------------------- #
# Algorithm 1: Task 1 — BERT multi-label tag classifier
# --------------------------------------------------------------------------- #
def train_bert(cfg, loaders, num_tags, device):
    model = BertTagger(cfg["text"]["bert_model_name"], num_tags, cfg["text"]["freeze_bert_layers"],
                        cfg["model"]["dropout"]).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg["train"]["bert_lr"], weight_decay=cfg["train"]["weight_decay"])
    history = []

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(loaders["train"], desc=f"[BERT] epoch {epoch+1}"):
            batch = to_device(batch, device)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = bce_multilabel_loss(logits, batch["tags"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loaders["train"])
        telemetry = validation_tag_metrics(model, loaders["val"], device, "bert")
        history.append({"epoch": epoch + 1, "train_loss": avg_loss, **telemetry})
        print(f"epoch {epoch+1}: train_loss={avg_loss:.4f}, val_auc_pr={telemetry['val_auc_pr']:.4f}")

    save_checkpoint(model, cfg, "bert_tagger.pt")
    return model, history


# --------------------------------------------------------------------------- #
# Algorithm 2: Task 2 — GNN encoder on music segment/chord graph
# --------------------------------------------------------------------------- #
def train_gnn(cfg, loaders, num_tags, in_dim, device):
    model = GNNTagger(in_dim=in_dim, cfg=cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    history = []

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(loaders["train"], desc=f"[GNN] epoch {epoch+1}"):
            batch = to_device(batch, device)
            g = batch["graph"]
            logits = model(g.x, g.edge_index, g.batch)
            loss = bce_multilabel_loss(logits, batch["tags"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loaders["train"])
        telemetry = validation_tag_metrics(model, loaders["val"], device, "gnn")
        history.append({"epoch": epoch + 1, "train_loss": avg_loss, **telemetry})
        print(f"epoch {epoch+1}: train_loss={avg_loss:.4f}, val_auc_pr={telemetry['val_auc_pr']:.4f}")

    save_checkpoint(model, cfg, "gnn_tagger.pt")
    return model, history


# --------------------------------------------------------------------------- #
# Algorithm 3: Task 3 — GNN-BERT fusion for context understanding
# --------------------------------------------------------------------------- #
def train_fusion(cfg, loaders, num_tags, in_dim, device, fusion_mode="cross_attn"):
    model = GNNBertFusion(audio_in_dim=in_dim, cfg=cfg, fusion_mode=fusion_mode).to(device)
    # BERT sub-layers use a smaller LR than the freshly-initialized GNN/fusion heads
    bert_params = list(model.bert.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("bert.")]
    optimizer = AdamW([
        {"params": bert_params, "lr": cfg["train"]["bert_lr"]},
        {"params": other_params, "lr": cfg["train"]["lr"]},
    ], weight_decay=cfg["train"]["weight_decay"])
    history = []

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        epoch_loss, epoch_parts = 0.0, {"tags": 0.0, "valence": 0.0, "arousal": 0.0}
        for batch in tqdm(loaders["train"], desc=f"[Fusion:{fusion_mode}] epoch {epoch+1}"):
            batch = to_device(batch, device)
            g = batch["graph"]
            tag_logits, emotion_pred = model(g.x, g.edge_index, g.batch, batch["input_ids"], batch["attention_mask"])
            loss, parts = multitask_loss(
                tag_logits, batch["tags"], emotion_pred, batch["emotion"], batch["has_emotion"],
                alpha=cfg["train"]["emotion_loss_weight"], beta=cfg["train"]["emotion_loss_weight"],
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            for k in epoch_parts:
                epoch_parts[k] += parts[k]

        n = len(loaders["train"])
        telemetry = validation_tag_metrics(model, loaders["val"], device, "fusion")
        history.append({"epoch": epoch + 1, "train_loss": epoch_loss / n,
                         **{f"train_{k}": v / n for k, v in epoch_parts.items()}, **telemetry})
        print(f"epoch {epoch+1}: train_loss={epoch_loss/n:.4f}, val_auc_pr={telemetry['val_auc_pr']:.4f}")

    save_checkpoint(model, cfg, f"fusion_{fusion_mode}.pt")
    return model, history


# --------------------------------------------------------------------------- #
# Algorithm 4: Task 4 — Contrastive GNN-BERT (MusicCaps)
# --------------------------------------------------------------------------- #
def train_contrastive(cfg, loaders, in_dim, device):
    model = DualEncoder(audio_in_dim=in_dim, cfg=cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    history = []

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(loaders["train"], desc=f"[Contrastive] epoch {epoch+1}"):
            batch = to_device(batch, device)
            g = batch["graph"]
            graph_emb, text_emb = model(g.x, g.edge_index, g.batch, batch["input_ids"], batch["attention_mask"])
            loss = info_nce_loss(graph_emb, text_emb, temperature=cfg["train"]["contrastive_temperature"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loaders["train"])
        telemetry = validation_retrieval_metrics(model, loaders["val"], device, cfg["train"]["contrastive_temperature"])
        history.append({"epoch": epoch + 1, "train_loss": avg_loss, **telemetry})
        print(f"epoch {epoch+1}: train_loss={avg_loss:.4f}, val_R@10={telemetry['val_caption_to_audio_R@10']:.4f}")

    save_checkpoint(model, cfg, "contrastive_dual_encoder.pt")
    return model, history


def save_checkpoint(model, cfg, filename):
    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, filename))


def save_history(history, cfg, name):
    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    path = os.path.join(cfg["paths"]["results_dir"], f"{name}_history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    plot_history(history, cfg, name)


def plot_history(history, cfg, name):
    """Save loss and validation metric curves required by the assignment."""
    epochs = [row["epoch"] for row in history]
    plot_dir = os.path.join(cfg["paths"]["results_dir"], "plots")
    os.makedirs(plot_dir, exist_ok=True)
    metric_keys = [key for key in history[0] if key.startswith("val_")]
    figure, axes = plt.subplots(1, 2 if metric_keys else 1, figsize=(11, 4))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", color="#1f77b4")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    if metric_keys:
        for key in metric_keys:
            axes[1].plot(epochs, [row[key] for row in history], marker="o", label=key.replace("val_", ""))
        axes[1].set(title="Validation telemetry", xlabel="Epoch", ylabel="Score")
        axes[1].legend(fontsize=8)
    figure.suptitle(name.replace("_", " ").title())
    figure.tight_layout()
    figure.savefig(os.path.join(plot_dir, f"{name}_telemetry.png"), dpi=160, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["bert", "gnn", "fusion", "contrastive"], required=True)
    parser.add_argument("--fusion_mode", choices=["cross_attn", "concat", "gnn_only", "bert_only"], default="cross_attn")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])
    device = get_device(cfg["train"]["device"])
    print(f"Using device: {device}")

    loaders, num_tags = build_dataloaders(cfg, args.task)
    in_dim = cfg["audio"]["n_mels"]  # segment-similarity graphs use mel-pooled node features

    if args.task == "bert":
        model, history = train_bert(cfg, loaders, num_tags, device)
        save_history(history, cfg, "task1_bert")
    elif args.task == "gnn":
        model, history = train_gnn(cfg, loaders, num_tags, in_dim, device)
        save_history(history, cfg, "task2_gnn")
    elif args.task == "fusion":
        model, history = train_fusion(cfg, loaders, num_tags, in_dim, device, args.fusion_mode)
        save_history(history, cfg, f"task3_fusion_{args.fusion_mode}")
    else:
        model, history = train_contrastive(cfg, loaders, in_dim, device)
        save_history(history, cfg, "task4_contrastive")

    print(f"Trainable parameters: {count_parameters(model):,}")
    print("Training complete. Run src/evaluate.py --task", args.task, "for metrics/plots.")


if __name__ == "__main__":
    main()
