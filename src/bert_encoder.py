"""
bert_encoder.py — Task 1 (Easy): BERT baseline for music tag understanding.

"BERT" (Bidirectional Encoder Representations from Transformers) is a
pretrained language model that reads a whole sentence at once (not
left-to-right) to build a context-aware vector for each token.
"CLS token" is a special placeholder token BERT prepends to every input;
after encoding, its vector is used as a summary of the whole sequence.

Model (spec 4.1):
    t = BERT_CLS(X_text)
    y_hat_k = sigmoid(w_k^T t + b_k)      for each of K tags

Loss: per-tag binary cross-entropy, averaged over K tags (spec eq. L_BERT).
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class BertTagger(nn.Module):
    """Fine-tunable BERT/DistilBERT encoder + linear multi-label head."""

    def __init__(self, model_name: str, num_tags: int, freeze_layers: int = 0, dropout: float = 0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Freeze the first `freeze_layers` transformer blocks: cheaper fine-tuning,
        # keeps low-level syntax features fixed, only adapts higher-level semantics.
        if freeze_layers > 0 and hasattr(self.bert, "transformer"):
            for layer in self.bert.transformer.layer[:freeze_layers]:
                for p in layer.parameters():
                    p.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_tags)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return the CLS embedding t, shape [batch, hidden_size]. Reused by fusion_model.py."""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]  # CLS token is always position 0

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (pre-sigmoid), shape [batch, num_tags]."""
        t = self.encode(input_ids, attention_mask)
        return self.classifier(self.dropout(t))


def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_batch(tokenizer, texts: list, max_length: int) -> dict:
    """Tokenize a batch of raw strings (lyrics/tags/captions) into BERT inputs."""
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def bce_multilabel_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """L_BERT = -(1/K) * sum_k [y_k log(y_hat_k) + (1-y_k) log(1-y_hat_k)].
    BCEWithLogitsLoss applies sigmoid internally, matching the sigma(.) in the spec."""
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="mean")


if __name__ == "__main__":
    # smoke test: verify shapes line up before wiring into train.py
    from utils import load_config

    cfg = load_config()
    tok = get_tokenizer(cfg["text"]["bert_model_name"])
    model = BertTagger(cfg["text"]["bert_model_name"], cfg["model"]["num_tags"], cfg["text"]["freeze_bert_layers"])
    batch = tokenize_batch(tok, ["melancholic piano ballad", "upbeat 1960s jazz"], cfg["text"]["max_length"])
    logits = model(batch["input_ids"], batch["attention_mask"])
    print("logits shape:", logits.shape)  # expect [2, num_tags]
