"""
graph_builder.py — Preprocessing pipeline, step 3 of the spec:

  Chord-transition graph : nodes = unique chords, edges = observed
                            transitions weighted by transition count.
  Segment graph          : nodes = time segments, edges = temporal adjacency
                            + cosine similarity of chroma/MFCC > tau.

Both graph types are exported as torch_geometric.data.Data objects, saved to
data/processed/graphs/{track_id}.pt, and consumed directly by src/gnn_model.py.

"Node" = a single vertex in the graph (one chord, or one time segment).
"Edge" = a connection between two nodes, optionally carrying a weight.
"""
import os
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from utils import load_config

# 12 pitch classes x {major, minor} = 24 simplified chord templates (spec 4-graph vocab)
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHORD_TEMPLATES = {}
for i, root in enumerate(PITCH_CLASSES):
    major = np.zeros(12); major[[i, (i + 4) % 12, (i + 7) % 12]] = 1
    minor = np.zeros(12); minor[[i, (i + 3) % 12, (i + 7) % 12]] = 1
    CHORD_TEMPLATES[f"{root}maj"] = major
    CHORD_TEMPLATES[f"{root}min"] = minor
CHORD_NAMES = list(CHORD_TEMPLATES.keys())
CHORD_MATRIX = np.stack([CHORD_TEMPLATES[c] for c in CHORD_NAMES])  # [24, 12]


def frames_to_chords(chroma_segment: np.ndarray) -> list:
    """
    Cheap chord estimation: average chroma across a segment's frames, then
    nearest-neighbor match against the 24 major/minor templates via cosine
    similarity. Good enough for building a *graph structure*; a real MIR
    pipeline would use a dedicated chord recognizer (e.g. madmom).
    """
    chords = []
    for frame_block in chroma_segment:
        mean_chroma = frame_block.mean(axis=0, keepdims=True)  # [1, 12]
        sims = cosine_similarity(mean_chroma, CHORD_MATRIX)[0]
        chords.append(CHORD_NAMES[int(np.argmax(sims))])
    return chords


def build_chord_transition_graph(chroma_segments: np.ndarray) -> Data:
    """
    Nodes = the 24 possible chords (fixed vocabulary, so node identity is
    shared across all tracks — this lets the GNN learn chord-general
    transition semantics rather than per-track ones).
    Edge weight[i, j] = count of chord_i -> chord_j transitions in this track.
    Node features = one-hot chord template (12-d) padded to gnn hidden size
    by the model's input projection, not here.
    """
    chord_seq = frames_to_chords(chroma_segments)
    n_chords = len(CHORD_NAMES)
    adj = np.zeros((n_chords, n_chords), dtype=np.float32)
    for a, b in zip(chord_seq[:-1], chord_seq[1:]):
        i, j = CHORD_NAMES.index(a), CHORD_NAMES.index(b)
        adj[i, j] += 1.0

    edge_index, edge_weight = [], []
    for i in range(n_chords):
        for j in range(n_chords):
            if adj[i, j] > 0:
                edge_index.append([i, j])
                edge_weight.append(adj[i, j])

    x = torch.tensor(CHORD_MATRIX, dtype=torch.float32)  # [24, 12] node features
    if len(edge_index) == 0:
        # degenerate silent/atonal track: self-loops only, so message passing is still valid
        edge_index = [[i, i] for i in range(n_chords)]
        edge_weight = [1.0] * n_chords
    edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight_t = torch.tensor(edge_weight, dtype=torch.float32)
    # normalize edge weights by max transition count so scale doesn't vary wildly across tracks
    edge_weight_t = edge_weight_t / edge_weight_t.max().clamp(min=1.0)

    return Data(x=x, edge_index=edge_index_t, edge_attr=edge_weight_t)


def build_segment_similarity_graph(mel_segments: np.ndarray, threshold: float, max_nodes: int) -> Data:
    """
    Nodes = time segments (capped at max_nodes for memory/compute).
    Edges = (a) temporal adjacency: segment i <-> i+1, always connected, and
            (b) similarity edges: cosine similarity of mean-pooled log-mel
                features between any two segments exceeds `threshold`.
    Node features = mean log-mel vector per segment, [n_segments, n_mels].
    """
    n_segments = max(1, min(mel_segments.shape[0], max_nodes))
    mel_segments = mel_segments[:n_segments]
    node_feats = mel_segments.mean(axis=1)  # [n_segments, n_mels] — average over frames in segment

    sims = cosine_similarity(node_feats)
    edge_index, edge_weight = [], []
    for i in range(n_segments):
        for j in range(n_segments):
            if i == j:
                continue
            is_temporal = abs(i - j) == 1
            is_similar = sims[i, j] > threshold
            if is_temporal or is_similar:
                edge_index.append([i, j])
                edge_weight.append(float(sims[i, j]))

    if not edge_index:
        edge_index = [[0, 0]]
        edge_weight = [1.0]
    x = torch.tensor(node_feats, dtype=torch.float32)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight_t = torch.tensor(edge_weight, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index_t, edge_attr=edge_weight_t)


def build_graphs_for_cache(feature_dir: str, out_dir: str, cfg: dict, graph_type: str = "segment") -> None:
    """
    graph_type: "chord" (Task 2 chord-transition variant) or
                "segment" (Task 2 segment-similarity variant, used by default
                in fusion/contrastive tasks per the spec's recommended pairing).
    """
    os.makedirs(out_dir, exist_ok=True)
    npz_files = [f for f in os.listdir(feature_dir) if f.endswith(".npz")]
    print(f"Building '{graph_type}' graphs for {len(npz_files)} cached tracks")

    for fn in tqdm(npz_files, desc="Building graphs"):
        track_id = os.path.splitext(fn)[0]
        out_path = os.path.join(out_dir, f"{track_id}.pt")
        if os.path.exists(out_path):
            continue
        data = np.load(os.path.join(feature_dir, fn))
        try:
            if graph_type == "chord":
                graph = build_chord_transition_graph(data["chroma"])
            else:
                graph = build_segment_similarity_graph(
                    data["mel"],
                    cfg["graph"]["similarity_threshold"],
                    cfg["graph"]["max_nodes_per_track"],
                )
            torch.save(graph, out_path)
        except Exception as e:
            print(f"  [skip] {track_id}: {e}")


if __name__ == "__main__":
    cfg = load_config()
    feature_dir = os.path.join(cfg["paths"]["processed_dir"], "features")
    graph_dir = os.path.join(cfg["paths"]["processed_dir"], "graphs")
    build_graphs_for_cache(feature_dir, graph_dir, cfg, graph_type="segment")
