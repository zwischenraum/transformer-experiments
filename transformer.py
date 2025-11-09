import torch
import torch.nn as nn
import math
from transformers import AutoTokenizer


class TransformerConfig:
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        vocab_size: int,
        dropout: float = 0.1,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.dropout = dropout

    def to_json(self):
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "n_layers": self.n_layers,
            "vocab_size": self.vocab_size,
            "dropout": self.dropout,
        }


def build_causal_mask(attention_mask: torch.Tensor):
    """
    attention_mask: padding mask of shape (batch_size, seq_len) where:
              - 1 indicates valid tokens
              - 0 indicates padding tokens

    Returns: causal mask of shape (batch_size, 1, seq_len, seq_len)
    """
    seq_len = attention_mask.shape[-1]

    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=attention_mask.device), diagonal=1
    ).bool()
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(
        0
    )  # Shape: (1, 1, seq_len, seq_len)

    padding_mask = (
        (1 - attention_mask).bool().unsqueeze(1).unsqueeze(1).repeat(1, 1, seq_len, 1)
    )  # shape: (batch_size, 1, seq_len, seq_len)

    return padding_mask | causal_mask  # shape: (batch_size, 1, seq_len, seq_len)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads

        self.Q = nn.Linear(
            in_features=self.d_model, out_features=self.d_k * self.n_heads
        )
        self.K = nn.Linear(
            in_features=self.d_model, out_features=self.d_k * self.n_heads
        )
        self.V = nn.Linear(
            in_features=self.d_model, out_features=self.d_k * self.n_heads
        )

        self.O = nn.Linear(
            in_features=self.d_k * self.n_heads, out_features=self.d_model
        )

        self.dropout = nn.Dropout(config.dropout)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_heads, seq_len, d_k = k.shape
        theta = 10000 ** (-2.0 / d_k * torch.arange(d_k // 2).to(q.device))
        positions = torch.arange(seq_len).to(q.device)
        angles = torch.outer(positions, theta)
        cos = angles.cos().repeat(1, 2)
        sin = angles.sin().repeat(1, 2)
        q = q * cos + self.rotate_half(q) * sin
        k = k * cos + self.rotate_half(k) * sin
        return q, k

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: input tensor of shape (batch_size, seq_len, d_model)
        mask: optional causal mask of shape (batch_size, 1, seq_len, seq_len) where:
              - 1 indicates valid tokens
              - 0 indicates padding tokens

        Returns: output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        q_proj = (
            self.Q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        )
        k_proj = (
            self.K(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        )
        v_proj = (
            self.V(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        )

        q_proj, k_proj = self._apply_rope(q_proj, k_proj)

        att_scores = (
            q_proj @ torch.transpose(k_proj, -2, -1) / math.sqrt(self.d_k)
        )  # shape (batch_size, n_heads, seq_len, seq_len).

        if mask is not None:
            att_scores = att_scores.masked_fill(mask, -torch.inf)

        att_probs = torch.softmax(att_scores, dim=-1)
        att_probs = self.dropout(att_probs)

        attn = att_probs @ v_proj  # shape (batch_size, n_heads, seq_len, d_k)
        o = (
            attn.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.n_heads * self.d_k)
        )

        return self.O(o)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ff_up = nn.Linear(in_features=config.d_model, out_features=config.d_ff)
        self.ff_down = nn.Linear(in_features=config.d_ff, out_features=config.d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: input tensor of shape (batch_size, seq_len, d_model)
        Returns: output tensor of shape (batch_size, seq_len, d_model)
        """
        h = self.ff_up(x)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.ff_down(h)
        return h


class DecoderBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_ff = config.d_ff

        self.mha = MultiHeadAttention(config)
        self.mlp = PositionwiseFeedForward(config)

        self.post_att_norm = nn.LayerNorm(self.d_model)
        self.post_mlp_norm = nn.LayerNorm(self.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: input tensor of shape (batch_size, seq_len, d_model)
        mask: causal mask of shape (batch_size, 1, seq_len, seq_len) where:
              - True indicates masked tokens
              - False indicates unmasked tokens

        Returns: output tensor of shape (batch_size, seq_len, d_model)
        """
        residual = x

        h = self.mha(x, mask)
        h = self.post_att_norm(h)
        h += residual

        residual = h

        h = self.mlp(h)
        h = self.post_mlp_norm(h)
        h += residual

        return h


class TransformerDecoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_ff = config.d_ff
        self.n_layers = config.n_layers
        self.dropout = config.dropout

        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=self.d_model
        )
        self.layers = nn.ModuleList(
            [DecoderBlock(config) for _ in range(self.n_layers)]
        )
        self.norm = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Parameter(self.token_embedding.weight.T)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the transformer decoder.

        Args:
            x: input token IDs of shape (batch_size, seq_len)
            mask: optional padding mask of shape (batch_size, seq_len) where:
                  - 1 indicates valid tokens
                  - 0 indicates padding tokens

        Returns:
            logits: output logits of shape (batch_size, seq_len, vocab_size)
        """
        device = x.device

        causal_mask = build_causal_mask(mask).to(device)
        x = self.token_embedding(x)
        for layer in self.layers:
            x = layer(x, causal_mask)
        return self.norm(x) @ self.lm_head.to(
            device
        )  # shape (batch_size, seq_len, vocab_size)


def main():
    d_model = 512
    n_heads = 8
    d_ff = 4 * d_model
    n_layers = 6

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Example with multiple sequences of different lengths
    texts = [
        "Hello, how are you?",
        "I am doing great!",
        "This is a longer sentence to demonstrate padding.",
    ]
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    tokens = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)  # This is your padding mask!

    vocab_size = tokenizer.vocab_size

    transformer_config = TransformerConfig(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        vocab_size=vocab_size,
    )
    model = TransformerDecoder(transformer_config)
    model.to(device)

    logits = model(tokens, attention_mask)
    print(f"Input shape: {tokens.shape}")
    print(f"Mask shape: {attention_mask.shape}")
    print(f"Output shape: {logits.shape}")


if __name__ == "__main__":
    main()
