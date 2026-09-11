"""
gnn_model.py — Task 2 (Medium): GNN on music structure graphs.

"GNN" (Graph Neural Network) is a model that updates each node's vector by
repeatedly aggregating information from its directly connected neighbors
("message passing"), so a node's final representation encodes both its own
features and its local graph context.
"GraphSAGE" ("SAmple and aggreGatE") updates a node by concatenating its own
previous vector with the mean of its neighbors' vectors, then applying a
learned linear layer.
"GAT" (Graph Attention Network) instead learns an attention weight per edge,
so a node can weigh some neighbors more than others.
"Readout" is the operation that pools all the node vectors in a graph into a
single graph-level vector.

Model (spec 4.2):
    h_i^(l+1) = sigma( W^(l) . CONCAT( h_i^(l), MEAN_{j in N(i)} h_j^(l) ) )
    g = MEAN_{i in V} h_i^(L)                       (mean-pool readout)
    y_hat = sigmoid(W g + b)

Also includes CNNBaseline: a small 2D-conv net on raw mel-spectrograms, used
as the "no graph, no text" comparison point (B2 in the spec's baseline list).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool


class GNNEncoder(nn.Module):
    """Stacks `num_layers` SAGEConv or GATConv blocks, then mean-pools to a graph vector g."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int,
                 gnn_type: str = "sage", gat_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.gnn_type = gnn_type
        self.convs = nn.ModuleList()

        def make_layer(d_in, d_out, is_last):
            if gnn_type == "gat":
                heads = 1 if is_last else gat_heads
                concat = False if is_last else True
                return GATConv(d_in, d_out // heads if concat and not is_last else d_out,
                                heads=heads, concat=concat, dropout=dropout)
            return SAGEConv(d_in, d_out)  # GraphSAGE: CONCAT(self, mean(neighbors)) + linear, built into SAGEConv

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            self.convs.append(make_layer(dims[i], dims[i + 1], is_last=(i == num_layers - 1)))

        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        """
        x: [total_nodes_in_batch, in_dim]
        edge_index: [2, total_edges_in_batch]
        batch: [total_nodes_in_batch] — maps each node to its graph index (PyG batching convention)
        Returns: node embeddings h [total_nodes, out_dim] AND pooled graph vector g [num_graphs, out_dim]
        """
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        g = global_mean_pool(h, batch)  # spec's g = (1/|V|) * sum_i h_i^(L)
        return h, g


class GNNTagger(nn.Module):
    """Task 2 end-to-end: GNNEncoder + linear multi-label head, y_hat = sigmoid(W g + b)."""

    def __init__(self, in_dim: int, cfg: dict):
        super().__init__()
        m = cfg["model"]
        self.encoder = GNNEncoder(
            in_dim=in_dim,
            hidden_dim=m["gnn_hidden_dim"],
            out_dim=m["gnn_out_dim"],
            num_layers=m["gnn_layers"],
            gnn_type=m["gnn_type"],
            gat_heads=m["gat_heads"],
            dropout=m["dropout"],
        )
        self.classifier = nn.Linear(m["gnn_out_dim"], m["num_tags"])

    def forward(self, x, edge_index, batch):
        _, g = self.encoder(x, edge_index, batch)
        return self.classifier(g)  # logits, [num_graphs, num_tags]


class CNNBaseline(nn.Module):
    """
    B2 baseline: plain 2D-CNN over a fixed-size log-mel spectrogram, no graph
    structure and no text. Exists purely to quantify how much the graph
    structure and BERT text actually help (Table 3 in the spec).
    """

    def __init__(self, n_mels: int, num_tags: int, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128 * 4 * 4, num_tags)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [batch, 1, n_frames, n_mels] -> logits [batch, num_tags]"""
        h = self.conv(mel)
        h = h.flatten(1)
        return self.fc(self.dropout(h))


if __name__ == "__main__":
    from utils import load_config
    cfg = load_config()
    model = GNNTagger(in_dim=cfg["audio"]["n_mels"], cfg=cfg)
    x = torch.randn(50, cfg["audio"]["n_mels"])
    edge_index = torch.randint(0, 50, (2, 200))
    batch = torch.zeros(50, dtype=torch.long)
    logits = model(x, edge_index, batch)
    print("GNN logits shape:", logits.shape)  # expect [1, num_tags]
