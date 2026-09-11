"""
fusion_model.py — Task 3 (Hard): GNN-BERT fusion for multi-context understanding.

"Cross-attention" lets one sequence (here, the graph's pooled vector, treated
as a length-1 "query") look up relevant information from another sequence
(the BERT token embeddings, treated as "keys"/"values") and pull out a
weighted combination of it. "Ablation" means removing one component of a
model to measure how much it was contributing.

Model (spec 4.3):
    A = softmax( Q K^T / sqrt(d) ),   Q = g W_Q,   K = H_text W_K
    z = CONCAT(g, A H_text)
    y_hat = sigmoid(W z)

Multi-task loss (spec eq.):
    L = L_tags + alpha * ||v - v_hat||^2 + beta * ||a - a_hat||^2
    (v, a) = DEAM valence/arousal targets, when available for a track.

Also implements the three ablation variants the spec asks for in Task 3's
deliverables: BERT-only, GNN-only, early-concat, cross-attention (default).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_encoder import BertTagger
from gnn_model import GNNEncoder


class CrossAttentionFusion(nn.Module):
    """Single-head cross-attention: graph vector g attends over BERT token embeddings."""

    def __init__(self, graph_dim: int, text_dim: int, attn_dim: int):
        super().__init__()
        self.W_Q = nn.Linear(graph_dim, attn_dim)
        self.W_K = nn.Linear(text_dim, attn_dim)
        self.W_V = nn.Linear(text_dim, attn_dim)
        self.scale = attn_dim ** 0.5

    def forward(self, g: torch.Tensor, H_text: torch.Tensor) -> torch.Tensor:
        """
        g:      [batch, graph_dim]           — one query vector per example
        H_text: [batch, seq_len, text_dim]   — full token sequence, not just CLS
        Returns attended context, [batch, attn_dim]
        """
        Q = self.W_Q(g).unsqueeze(1)             # [batch, 1, attn_dim]
        K = self.W_K(H_text)                     # [batch, seq_len, attn_dim]
        V = self.W_V(H_text)                     # [batch, seq_len, attn_dim]
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [batch, 1, seq_len]
        A = F.softmax(scores, dim=-1)
        context = torch.bmm(A, V).squeeze(1)     # [batch, attn_dim]
        return context


class GNNBertFusion(nn.Module):
    """
    End-to-end Task 3 model. `fusion_mode` selects which of the four spec
    ablations to run, all sharing the same GNN/BERT backbones:
      - "cross_attn" (default): spec's recommended fusion
      - "concat"               : z = CONCAT(g, t)  (early concat, CLS only)
      - "gnn_only"              : z = g, BERT branch unused
      - "bert_only"             : z = t, GNN branch unused
    """

    def __init__(self, audio_in_dim: int, cfg: dict, fusion_mode: str = "cross_attn"):
        super().__init__()
        self.fusion_mode = fusion_mode
        m, t = cfg["model"], cfg["text"]

        self.gnn = GNNEncoder(
            in_dim=audio_in_dim, hidden_dim=m["gnn_hidden_dim"], out_dim=m["gnn_out_dim"],
            num_layers=m["gnn_layers"], gnn_type=m["gnn_type"], gat_heads=m["gat_heads"], dropout=m["dropout"],
        )
        self.bert = BertTagger(t["bert_model_name"], m["num_tags"], t["freeze_bert_layers"], m["dropout"])
        text_dim = self.bert.bert.config.hidden_size

        self.cross_attn = CrossAttentionFusion(m["gnn_out_dim"], text_dim, m["fusion_dim"])

        fused_dim = {
            "cross_attn": m["gnn_out_dim"] + m["fusion_dim"],
            "concat": m["gnn_out_dim"] + text_dim,
            "gnn_only": m["gnn_out_dim"],
            "bert_only": text_dim,
        }[fusion_mode]

        self.dropout = nn.Dropout(m["dropout"])
        self.tag_head = nn.Linear(fused_dim, m["num_tags"])
        self.emotion_head = nn.Linear(fused_dim, 2)  # [valence, arousal]

    def forward(self, x, edge_index, batch, input_ids, attention_mask):
        _, g = self.gnn(x, edge_index, batch)
        bert_out = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        H_text = bert_out.last_hidden_state       # [batch, seq_len, text_dim]
        t = H_text[:, 0, :]                       # CLS vector

        if self.fusion_mode == "cross_attn":
            context = self.cross_attn(g, H_text)
            z = torch.cat([g, context], dim=-1)
        elif self.fusion_mode == "concat":
            z = torch.cat([g, t], dim=-1)
        elif self.fusion_mode == "gnn_only":
            z = g
        else:  # bert_only
            z = t

        z = self.dropout(z)
        tag_logits = self.tag_head(z)
        emotion_pred = self.emotion_head(z)  # [batch, 2] -> (valence_hat, arousal_hat)
        return tag_logits, emotion_pred


def multitask_loss(tag_logits, tag_targets, emotion_pred, emotion_targets, emotion_mask, alpha, beta):
    """
    L = L_tags + alpha*||v - v_hat||^2 + beta*||a - a_hat||^2

    emotion_mask: [batch] boolean — True where a DEAM valence/arousal label
    exists for that example (not every track has emotion annotations), so the
    regression terms only backprop through examples that actually have them.
    """
    l_tags = F.binary_cross_entropy_with_logits(tag_logits, tag_targets)

    if emotion_mask.any():
        v_hat, a_hat = emotion_pred[emotion_mask, 0], emotion_pred[emotion_mask, 1]
        v, a = emotion_targets[emotion_mask, 0], emotion_targets[emotion_mask, 1]
        l_valence = F.mse_loss(v_hat, v)
        l_arousal = F.mse_loss(a_hat, a)
    else:
        l_valence = torch.tensor(0.0, device=tag_logits.device)
        l_arousal = torch.tensor(0.0, device=tag_logits.device)

    return l_tags + alpha * l_valence + beta * l_arousal, {
        "tags": l_tags.item(), "valence": l_valence.item(), "arousal": l_arousal.item()
    }
