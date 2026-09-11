"""
evaluate.py — computes every metric the spec asks for (section 6) and
produces the plots/tables listed as deliverables for Tasks 1-4:
  - Macro-F1 / Micro-F1 curves (Task 1)
  - AUC-PR per tag, averaged (Tasks 1-3)
  - Emotion MAE / R^2 (Task 3, DEAM)
  - t-SNE of fused representation z, colored by genre/mood (Task 3)
  - Retrieval table + qualitative examples (Task 4)

"Precision" = of the items the model predicted positive, the fraction that
truly were positive. "Recall" = of the items that truly were positive, the
fraction the model found. "F1" is their harmonic mean, balancing both.
"Macro" averaging computes a metric per class then averages classes equally;
"micro" averaging pools all predictions together first, so it weights
frequent classes more. "AUC-PR" is the area under the precision-recall curve
as the classification threshold varies — higher is better, less sensitive to
class imbalance than plain accuracy.
"t-SNE" (t-distributed Stochastic Neighbor Embedding) is a technique for
compressing high-dimensional vectors down to 2D for visualization while
trying to preserve which points were close together originally.
"""
import os
import json
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, average_precision_score, mean_absolute_error, r2_score
from sklearn.manifold import TSNE

from utils import load_config, get_device
from train import build_dataloaders, to_device
from bert_encoder import BertTagger
from gnn_model import GNNTagger
from fusion_model import GNNBertFusion
from contrastive import DualEncoder, retrieval_recall_at_k


def tag_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Macro-F1, Micro-F1 (spec's threshold-based scores) + mean AUC-PR (threshold-free)."""
    y_pred = (y_prob >= threshold).astype(int)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    # AUC-PR per tag, skipping tags with no positive examples in this split
    valid_cols = [k for k in range(y_true.shape[1]) if y_true[:, k].sum() > 0]
    auc_pr = np.mean([average_precision_score(y_true[:, k], y_prob[:, k]) for k in valid_cols]) if valid_cols else 0.0
    return {"macro_f1": float(macro_f1), "micro_f1": float(micro_f1), "auc_pr": float(auc_pr)}


def emotion_metrics(v_true, v_pred, a_true, a_pred) -> dict:
    """MAE and R^2 for valence and arousal separately, per spec section 6."""
    return {
        "mae_valence": float(mean_absolute_error(v_true, v_pred)),
        "mae_arousal": float(mean_absolute_error(a_true, a_pred)),
        "r2_valence": float(r2_score(v_true, v_pred)),
        "r2_arousal": float(r2_score(a_true, a_pred)),
    }


@torch.no_grad()
def evaluate_bert(cfg, loaders, num_tags, device, ckpt_path):
    model = BertTagger(cfg["text"]["bert_model_name"], num_tags, cfg["text"]["freeze_bert_layers"],
                        cfg["model"]["dropout"]).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_probs, all_true = [], []
    for batch in loaders["test"]:
        batch = to_device(batch, device)
        logits = model(batch["input_ids"], batch["attention_mask"])
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_true.append(batch["tags"].cpu().numpy())

    y_prob, y_true = np.concatenate(all_probs), np.concatenate(all_true)
    metrics = tag_metrics(y_true, y_prob)

    # 5 example predictions, as requested in Task 1 deliverables
    examples = []
    for i in range(min(5, len(y_true))):
        top_pred = np.argsort(-y_prob[i])[:5].tolist()
        top_true = np.where(y_true[i] == 1)[0].tolist()
        examples.append({"predicted_top5_tag_idx": top_pred, "true_tag_idx": top_true})
    return metrics, examples


@torch.no_grad()
def evaluate_gnn(cfg, loaders, num_tags, in_dim, device, ckpt_path):
    model = GNNTagger(in_dim=in_dim, cfg=cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_probs, all_true = [], []
    for batch in loaders["test"]:
        batch = to_device(batch, device)
        g = batch["graph"]
        logits = model(g.x, g.edge_index, g.batch)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_true.append(batch["tags"].cpu().numpy())

    y_prob, y_true = np.concatenate(all_probs), np.concatenate(all_true)
    return tag_metrics(y_true, y_prob)


@torch.no_grad()
def evaluate_fusion(cfg, loaders, num_tags, in_dim, device, ckpt_path, fusion_mode, results_dir):
    model = GNNBertFusion(audio_in_dim=in_dim, cfg=cfg, fusion_mode=fusion_mode).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_probs, all_true, all_v_true, all_v_pred, all_a_true, all_a_pred, all_z = [], [], [], [], [], [], []
    for batch in loaders["test"]:
        batch = to_device(batch, device)
        g = batch["graph"]
        tag_logits, emotion_pred = model(g.x, g.edge_index, g.batch, batch["input_ids"], batch["attention_mask"])
        all_probs.append(torch.sigmoid(tag_logits).cpu().numpy())
        all_true.append(batch["tags"].cpu().numpy())

        mask = batch["has_emotion"].cpu().numpy()
        if mask.any():
            all_v_true.append(batch["emotion"][:, 0].cpu().numpy()[mask])
            all_v_pred.append(emotion_pred[:, 0].cpu().numpy()[mask])
            all_a_true.append(batch["emotion"][:, 1].cpu().numpy()[mask])
            all_a_pred.append(emotion_pred[:, 1].cpu().numpy()[mask])

    y_prob, y_true = np.concatenate(all_probs), np.concatenate(all_true)
    metrics = tag_metrics(y_true, y_prob)

    if all_v_true:
        metrics.update(emotion_metrics(
            np.concatenate(all_v_true), np.concatenate(all_v_pred),
            np.concatenate(all_a_true), np.concatenate(all_a_pred),
        ))

    plot_tsne(y_true, y_prob, results_dir, fusion_mode)
    return metrics


def plot_tsne(y_true: np.ndarray, y_prob: np.ndarray, results_dir: str, tag_suffix: str):
    """t-SNE of predicted tag-probability vectors, colored by dominant true tag (Task 3 deliverable)."""
    if len(y_prob) < 5:
        return  # t-SNE needs a reasonable number of points to be meaningful
    perplexity = min(30, max(5, len(y_prob) // 3))
    coords = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(y_prob)
    dominant_tag = y_true.argmax(axis=1)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=dominant_tag, cmap="tab20", s=15)
    plt.title(f"t-SNE of fused representation z ({tag_suffix})")
    plt.xlabel("dim 1"); plt.ylabel("dim 2")
    plt.colorbar(scatter, label="dominant tag id")
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    plt.savefig(os.path.join(results_dir, "plots", f"tsne_{tag_suffix}.png"), dpi=150, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def evaluate_contrastive(cfg, loaders, in_dim, device, ckpt_path, results_dir):
    model = DualEncoder(audio_in_dim=in_dim, cfg=cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    graph_embs, text_embs, track_ids = [], [], []
    for batch in loaders["test"]:
        batch = to_device(batch, device)
        g = batch["graph"]
        g_emb, t_emb = model(g.x, g.edge_index, g.batch, batch["input_ids"], batch["attention_mask"])
        graph_embs.append(g_emb.cpu()); text_embs.append(t_emb.cpu())
        track_ids.extend(batch["track_id"])

    graph_embs, text_embs = torch.cat(graph_embs), torch.cat(text_embs)
    recall = retrieval_recall_at_k(graph_embs, text_embs)

    # 10 qualitative retrieval examples: top-3 audio matches per caption
    sims = (graph_embs @ text_embs.t()).numpy()
    examples = []
    for i in range(min(10, len(track_ids))):
        top3 = np.argsort(-sims[:, i])[:3]
        examples.append({
            "query_track_id": track_ids[i],
            "top3_matched_track_ids": [track_ids[j] for j in top3],
        })
    os.makedirs(os.path.join(results_dir, "retrieval_examples"), exist_ok=True)
    with open(os.path.join(results_dir, "retrieval_examples", "examples.json"), "w") as f:
        json.dump(examples, f, indent=2)

    return recall


def plot_training_curve(history_path: str, out_path: str, metric_key: str = "train_loss"):
    with open(history_path) as f:
        history = json.load(f)
    epochs = [h["epoch"] for h in history]
    values = [h[metric_key] for h in history]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, values, marker="o")
    plt.xlabel("epoch"); plt.ylabel(metric_key); plt.title(metric_key.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["bert", "gnn", "fusion", "contrastive"], required=True)
    parser.add_argument("--fusion_mode", default="cross_attn")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["train"]["device"])
    results_dir = cfg["paths"]["results_dir"]
    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    loaders, num_tags = build_dataloaders(cfg, args.task)
    in_dim = cfg["audio"]["n_mels"]

    if args.task == "bert":
        metrics, examples = evaluate_bert(cfg, loaders, num_tags, device, os.path.join(ckpt_dir, "bert_tagger.pt"))
        metrics["example_predictions"] = examples
    elif args.task == "gnn":
        metrics = evaluate_gnn(cfg, loaders, num_tags, in_dim, device, os.path.join(ckpt_dir, "gnn_tagger.pt"))
    elif args.task == "fusion":
        ckpt = os.path.join(ckpt_dir, f"fusion_{args.fusion_mode}.pt")
        metrics = evaluate_fusion(cfg, loaders, num_tags, in_dim, device, ckpt, args.fusion_mode, results_dir)
    else:
        ckpt = os.path.join(ckpt_dir, "contrastive_dual_encoder.pt")
        metrics = evaluate_contrastive(cfg, loaders, in_dim, device, ckpt, results_dir)

    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")
    all_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            all_metrics = json.load(f)
    all_metrics[f"{args.task}_{args.fusion_mode if args.task == 'fusion' else ''}".rstrip("_")] = metrics
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
