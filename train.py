import torch
import torch.nn.functional as F
from transformer import TransformerConfig, TransformerDecoder


def main():
    examples = 1000
    seq_len = 10
    vocab_size = 100
    d_model = 512
    n_heads = 8
    d_ff = 4 * d_model
    n_layers = 6

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")

    transformer_config = TransformerConfig(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        vocab_size=vocab_size,
    )
    model = TransformerDecoder(transformer_config)
    model.to(device)
    data = torch.randint(0, vocab_size, (examples, seq_len)).to(device)
    input_ids = data[:, :-1].to(device)
    targets = data[:, 1:].to(device)
    mask = torch.triu(torch.ones(seq_len - 1, seq_len - 1), diagonal=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(examples):
        logits = model(input_ids[i].unsqueeze(0), mask)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets[i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())


if __name__ == "__main__":
    main()
