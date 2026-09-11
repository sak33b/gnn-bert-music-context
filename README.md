# GNN-BERT Music Context Understanding

Implementation details of all four tasks from the CSE425 Project: BERT tag
classification, GraphSAGE/GAT structure graphs, cross-attention GNN-BERT
fusion, and contrastive audio-caption retrieval.

## 1. Setup

```bash
uv venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -r requirements.txt
```

## 2. Data layout

Place raw downloads under `data/raw/` exactly as documented at the top of
`config.yaml`:

```
data/raw/
  fma_medium/                 FMA-medium mp3s (fma's own 000/xxx.mp3 layout)
  fma_metadata/tracks.csv
  magnatagatune/mp3/ + annotations_final.csv
  gtzan/genres_original/
  deam/audio/ + deam/annotations/static_annotations.csv
  musiccaps/audio/{ytid}.wav + musiccaps/musiccaps-public.csv
```


## 3. Pipeline (run in order)

```bash
python src/audio_features.py

python src/graph_builder.py

python src/prepare_musiccaps.py

```

## 4. Training (spec Algorithms 1-4)

```bash
python src/train.py --task bert                              # Task 1
python src/train.py --task gnn                                # Task 2
python src/train.py --task fusion --fusion_mode cross_attn    # Task 3 (default fusion)
python src/train.py --task fusion --fusion_mode concat        # Task 3 ablation
python src/train.py --task fusion --fusion_mode gnn_only      # Task 3 ablation
python src/train.py --task fusion --fusion_mode bert_only     # Task 3 ablation
python src/train.py --task contrastive                        # Task 4

# Task 2's fair mel-spectrogram comparison (no graph, no text)
python src/run_cnn_baseline.py
```

Checkpoints land in `results/checkpoints/`; per-epoch loss history in
`results/<task>_history.json`. Each training epoch also records validation
Macro-F1, Micro-F1, and mean AUC-PR (or InfoNCE and bidirectional R@K for
contrastive retrieval), and writes a telemetry plot to `results/plots/`.

## 5. Evaluation (spec section 6 + Task 3/4 deliverables)

```bash
python src/evaluate.py --task bert
python src/evaluate.py --task gnn
python src/evaluate.py --task fusion --fusion_mode cross_attn
python src/evaluate.py --task contrastive
```

Produces:
- `results/metrics.json` — Macro-F1, Micro-F1, AUC-PR, emotion MAE/R², recall@K, merged across every task
- `results/plots/tsne_<fusion_mode>.png` — t-SNE of fused tag-probability space
- `results/retrieval_examples/examples.json` — 10 caption -> top-3 audio matches (Task 4)

## 6. Files in `src/`

| File | Task(s) | Role |
|---|---|---|
| `utils.py` | all | config loading, seeding, device selection |
| `audio_features.py` | preprocessing | resample, log-mel/chroma, normalize, segment |
| `graph_builder.py` | preprocessing | chord-transition & segment-similarity graph construction |
| `datasets.py` | all | PyTorch `Dataset`/`DataLoader` glue over graphs + tags + captions + DEAM labels |
| `bert_encoder.py` | 1, 3, 4 | BERT/DistilBERT wrapper + Task 1 tag classifier |
| `gnn_model.py` | 2, 3, 4 | GraphSAGE/GAT encoder, Task 2 tagger, CNN baseline |
| `fusion_model.py` | 3 | cross-attention fusion + 3 ablation variants + multi-task loss |
| `contrastive.py` | 4 | dual encoder, InfoNCE loss, retrieval R@K |
| `train.py` | all | single CLI entrypoint, one function per spec Algorithm 1-4 |
| `evaluate.py` | all | Macro/Micro-F1, AUC-PR, emotion MAE/R², t-SNE, retrieval table |
