"""
contrastive.py — Task 4 (Advanced): cross-modal MusicCaps alignment.

"Contrastive learning" trains a model to pull matching pairs (a clip and its
correct caption) close together in embedding space while pushing unmatched
pairs apart, using only positive/negative pairing as supervision.
"InfoNCE" ("Noise-Contrastive Estimation") is the specific loss used here: for
each anchor, treat its true pair as the one positive among many negatives
sampled from the same batch, and maximize the ratio of positive to total
similarity via a softmax-style log loss.
"Dual encoder" means the two modalities (graph, text) are each encoded by
their own separate network, and only their final vectors are compared.

Loss (spec section 4.4):
    L_NCE = -log( exp(sim(g_i,t_i)/tau) / sum_j exp(sim(g_i,t_j)/tau) )
    sim(u,v) = u^T v / (||u|| ||v||)          (cosine similarity)

Retrieval metrics: Caption->Audio R@1/R@5/R@10 and Audio->Caption R@K, i.e.
"is the true match ranked in the top K by similarity".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_encoder import BertTagger
from gnn_model import GNNEncoder


class DualEncoder(nn.Module):
    """Separate GNN and BERT towers, each projected into a shared embedding space."""

    def __init__(self, audio_in_dim: int, cfg: dict):
        super().__init__()
        m, t = cfg["model"], cfg["text"]
        self.gnn = GNNEncoder(
            in_dim=audio_in_dim, hidden_dim=m["gnn_hidden_dim"], out_dim=m["gnn_out_dim"],
            num_layers=m["gnn_layers"], gnn_type=m["gnn_type"], gat_heads=m["gat_heads"], dropout=m["dropout"],
        )
        self.bert = BertTagger(t["bert_model_name"], m["num_tags"], t["freeze_bert_layers"], m["dropout"])
        text_dim = self.bert.bert.config.hidden_size

        shared_dim = m["fusion_dim"]
        self.graph_proj = nn.Linear(m["gnn_out_dim"], shared_dim)
        self.text_proj = nn.Linear(text_dim, shared_dim)

    def encode_graph(self, x, edge_index, batch) -> torch.Tensor:
        _, g = self.gnn(x, edge_index, batch)
        g = self.graph_proj(g)
        return F.normalize(g, dim=-1)  # L2-normalize so dot product == cosine similarity

    def encode_text(self, input_ids, attention_mask) -> torch.Tensor:
        t = self.bert.encode(input_ids, attention_mask)
        t = self.text_proj(t)
        return F.normalize(t, dim=-1)

    def forward(self, x, edge_index, batch, input_ids, attention_mask):
        return self.encode_graph(x, edge_index, batch), self.encode_text(input_ids, attention_mask)


def info_nce_loss(graph_emb: torch.Tensor, text_emb: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    graph_emb, text_emb: [batch, shared_dim], both L2-normalized, with
    matching row index i meaning (graph_i, text_i) is a true pair.
    Computes the loss symmetrically (graph->text and text->graph) and averages,
    which is standard practice (as in CLIP) and stabilizes training.
    """
    batch_size = graph_emb.size(0)
    logits = graph_emb @ text_emb.t() / temperature       # [batch, batch] similarity matrix S_ij
    labels = torch.arange(batch_size, device=graph_emb.device)

    loss_g2t = F.cross_entropy(logits, labels)             # caption retrieval given audio graph
    loss_t2g = F.cross_entropy(logits.t(), labels)          # audio retrieval given caption
    return (loss_g2t + loss_t2g) / 2.0


@torch.no_grad()
def retrieval_recall_at_k(graph_emb: torch.Tensor, text_emb: torch.Tensor, ks=(1, 5, 10)) -> dict:
    """
    Computes Caption->Audio and Audio->Caption R@K over a held-out batch/set.
    R@K = fraction of queries whose true match appears in the top-K most
    similar candidates.
    """
    sims = graph_emb @ text_emb.t()  # [N, N]
    n = sims.size(0)
    labels = torch.arange(n, device=sims.device)

    results = {}
    for direction, mat in [("audio_to_caption", sims), ("caption_to_audio", sims.t())]:
        ranks = mat.argsort(dim=1, descending=True)  # [N, N] candidate indices sorted by similarity
        for k in ks:
            topk = ranks[:, :k]
            hit = (topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()
            results[f"{direction}_R@{k}"] = hit
    return results


if __name__ == "__main__":
    # smoke test with random embeddings to sanity-check the loss/metric shapes
    torch.manual_seed(0)
    g = F.normalize(torch.randn(8, 256), dim=-1)
    t = F.normalize(torch.randn(8, 256), dim=-1)
    print("InfoNCE loss:", info_nce_loss(g, t, temperature=0.07).item())
    print("Recall@K:", retrieval_recall_at_k(g, t))
