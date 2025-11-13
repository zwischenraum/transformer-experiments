import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformer import TransformerConfig, TransformerDecoder
import wandb
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2TokenizerFast

wandb.init(project="transformer", name="train")


def create_dataset(seq_len: int, batch_size: int = 10):
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )
    tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len,
        )

    dataset = dataset.map(tokenize, batched=True, batch_size=batch_size)
    dataset.with_format("torch")
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    return dataloader, tokenizer


def main():
    num_steps = 50
    seq_len = 1024
    batch_size = 4
    d_model = 512
    n_heads = 8
    d_ff = 4 * d_model
    n_layers = 6

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")

    dataloader, tokenizer = create_dataset(seq_len=seq_len, batch_size=batch_size)
    vocab_size = tokenizer.vocab_size

    transformer_config = TransformerConfig(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        vocab_size=vocab_size,
    )
    wandb.config.update(transformer_config.to_json())
    model = TransformerDecoder(transformer_config)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    step = 0
    pbar = tqdm(dataloader, total=num_steps, desc="Loss: 0.0000")
    for batch in pbar:
        if step >= num_steps:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        targets = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        attention_mask = attention_mask[:, :-1].contiguous()

        logits = model(input_ids, attention_mask)

        loss_mask = attention_mask.view(-1) == 1
        logits_masked = logits.view(-1, vocab_size)[loss_mask]
        targets_masked = targets.view(-1)[loss_mask]

        loss = F.cross_entropy(logits_masked, targets_masked)

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()

        wandb.log({"loss": loss.item()}, step=step)

        pbar.set_description(f"Loss: {loss.item():.4f}")

        step += 1

    torch.save(model.state_dict(), "model.pth")
    wandb.save("model.pth")
    wandb.finish()


if __name__ == "__main__":
    main()
