from typing import Any
import torch
import torch.nn as nn


class MLPModel(nn.Module):
    """
    BoW regressor: [B, V] -> [B, 1]
    """

    def __init__(self, args, input_dim: int, output_dim: int = 1):
        super().__init__()
        hidden_dim = getattr(args, "hidden_dim", 256)
        dropout = getattr(args, "dropout", 0.2)
        n_layers = getattr(args, "n_layers", 2)

        layers = []
        d_in = input_dim

        if n_layers <= 1:
            layers.append(nn.Linear(d_in, output_dim))
        else:
            for _ in range(n_layers - 1):
                layers.append(nn.Linear(d_in, hidden_dim))
                layers.append(nn.ReLU())
                if dropout and dropout > 0:
                    layers.append(nn.Dropout(dropout))
                d_in = hidden_dim
            layers.append(nn.Linear(d_in, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EmbeddingModel(nn.Module):
    """
    input_ids [B, L] -> embeddings [B, L, D] -> pooling -> MLP -> [B, 1]
    pooling: mean | sum | max
    """

    def __init__(self, args, word2vec: Any, n_labels: int = 1):
        super().__init__()
        self.pad_idx = word2vec.get_index("[PAD]")

        vocab_size = len(word2vec.key_to_index) if hasattr(word2vec, "key_to_index") else word2vec.vectors.shape[0]
        emb_dim = word2vec.vector_size

        weights = torch.tensor(word2vec.vectors, dtype=torch.float32)

        self.embedding = nn.Embedding.from_pretrained(
            weights,
            freeze=not getattr(args, "train_embeddings", False),
            padding_idx=self.pad_idx,
        )

        self.pooling = getattr(args, "pooling", "mean").lower()  # mean|sum|max

        hidden_dim = getattr(args, "hidden_dim", 256)
        dropout = getattr(args, "dropout", 0.2)
        n_layers = getattr(args, "n_layers", 2)

        layers = []
        d_in = emb_dim

        if n_layers <= 1:
            layers.append(nn.Linear(d_in, n_labels))
        else:
            for _ in range(n_layers - 1):
                layers.append(nn.Linear(d_in, hidden_dim))
                layers.append(nn.ReLU())
                if dropout and dropout > 0:
                    layers.append(nn.Dropout(dropout))
                d_in = hidden_dim
            layers.append(nn.Linear(d_in, n_labels))

        self.head = nn.Sequential(*layers)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids)  # [B, L, D]
        mask = (input_ids != self.pad_idx).unsqueeze(-1)  # [B, L, 1]

        if self.pooling == "max":
            emb_masked = emb.masked_fill(~mask, -1e9)
            pooled = emb_masked.max(dim=1).values
        elif self.pooling == "sum":
            pooled = (emb * mask).sum(dim=1)
        elif self.pooling == "mean":
            emb_masked = emb * mask
            lengths = mask.sum(dim=1).clamp(min=1)
            pooled = emb_masked.sum(dim=1) / lengths
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return self.head(pooled)
