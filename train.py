import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformer import TransformerConfig, TransformerDecoder
import wandb
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import yaml
from typing import Any

RUN_ID_ENV = "WANDB_RUN_ID"
RESUME_ARTIFACT_ENV = "WANDB_RESUME_ARTIFACT"
CONFIG_PATH_ENV = "TRAIN_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path("config/training.yaml")


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found at {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config file root must be a mapping.")
    missing_sections = {"run", "training", "model", "dataset"} - set(config.keys())
    if missing_sections:
        raise KeyError(
            f"Config missing required sections: {', '.join(sorted(missing_sections))}"
        )
    return config


def create_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_config: dict[str, Any],
    seq_len: int,
    batch_size: int,
):
    dataset_path = dataset_config.get("path")
    dataset_split = dataset_config.get("split")
    if dataset_path is None or dataset_split is None:
        raise KeyError("Dataset config must define 'path' and 'split'.")
    dataset = load_dataset(
        dataset_path,
        name=dataset_config.get("name"),
        split=dataset_split,
        streaming=bool(dataset_config.get("streaming")),
    )
    shuffle_config = dataset_config.get("shuffle")
    if shuffle_config:
        if "seed" not in shuffle_config or "buffer_size" not in shuffle_config:
            raise KeyError("Shuffle config must define 'seed' and 'buffer_size'.")
        dataset = dataset.shuffle(
            int(shuffle_config["seed"]),
            buffer_size=int(shuffle_config["buffer_size"]),
        )

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len,
        )

    tokenization_batch_size = int(
        dataset_config.get("tokenizer_batch_size", batch_size)
    )
    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=tokenization_batch_size,
    )
    dataset = dataset.with_format("torch")
    return dataset


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    dataset_state: dict,
    model_config: dict,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "dataset": dataset_state,
            "model_config": model_config,
        },
        path,
    )


def main():
    config_path_str = os.environ.get(CONFIG_PATH_ENV)
    config_path = Path(config_path_str) if config_path_str else DEFAULT_CONFIG_PATH
    config = load_config(config_path)

    run_config = config["run"]
    training_config = config["training"]
    model_config_data = config["model"]
    dataset_config = config["dataset"]

    project = run_config.get("project")
    if not project:
        raise KeyError("Run config must define 'project'.")
    run_name = run_config.get("name", "train")

    artifact_config = run_config.get("artifact", {})
    artifact_name = artifact_config.get("name", "model-checkpoint")
    checkpoint_filename = artifact_config.get("filename", "checkpoint.pt")

    total_steps = int(training_config["total_steps"])
    seq_len = int(training_config["seq_len"])
    batch_size = int(training_config["batch_size"])
    lr = float(training_config["learning_rate"])

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")

    run_id = os.environ.get(RUN_ID_ENV) or run_config.get("id")
    resume_artifact = os.environ.get(RESUME_ARTIFACT_ENV) or run_config.get(
        "resume_artifact"
    )
    if run_id:
        wandb_run = wandb.init(
            project=project, name=run_name, id=run_id, resume="allow"
        )
    else:
        wandb_run = wandb.init(project=project, name=run_name)

    tokenizer_name = dataset_config.get("tokenizer")
    if not tokenizer_name:
        raise KeyError("Dataset config must define 'tokenizer'.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer must define either a pad_token or eos_token for padding."
            )
        tokenizer.pad_token = tokenizer.eos_token

    dataset = create_dataset(
        tokenizer=tokenizer,
        dataset_config=dataset_config,
        seq_len=seq_len,
        batch_size=batch_size,
    )

    n_layers = int(model_config_data["n_layers"])
    d_model = int(model_config_data["d_model"])
    n_heads = int(model_config_data["n_heads"])
    d_ff = int(model_config_data["d_ff"])
    dropout = float(model_config_data.get("dropout", 0.1))
    vocab_size = tokenizer.vocab_size

    transformer_config = TransformerConfig(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        vocab_size=vocab_size,
        dropout=dropout,
    )
    model_config = transformer_config.to_json()
    wandb_run.config.update(
        {
            "training": {
                "total_steps": total_steps,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "learning_rate": lr,
            },
            "model": model_config_data,
            "dataset": dataset_config,
        }
    )
    model = TransformerDecoder(transformer_config)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    start_step = 0
    if resume_artifact:
        artifact = wandb_run.use_artifact(resume_artifact, type="model")
        checkpoint_dir = Path(artifact.download())
        checkpoint_path = checkpoint_dir / checkpoint_filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Expected checkpoint at {checkpoint_path}, but it was not found."
            )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint["model_config"] != model_config:
            raise RuntimeError("Checkpoint model configuration does not match.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        dataset.load_state_dict(checkpoint["dataset"])
        start_step = int(checkpoint["step"])

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    step = start_step
    pbar = tqdm(dataloader, total=total_steps, initial=step, desc="Loss: 0.0000")
    for batch in pbar:
        if step >= total_steps:
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

        wandb_run.log({"loss": loss.item()}, step=step)

        pbar.set_description(f"Loss: {loss.item():.4f}")

        step += 1

    if step == start_step:
        raise RuntimeError(
            "No training steps were executed; increase total_steps to continue training."
        )

    dataset_state = dataset.state_dict()
    checkpoint_path = Path(checkpoint_filename)
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=step,
        dataset_state=dataset_state,
        model_config=model_config,
    )
    artifact = wandb.Artifact(artifact_name, type="model")
    artifact.add_file(str(checkpoint_path))
    artifact.metadata["step"] = step
    wandb_run.log_artifact(artifact, aliases=["latest"])
    wandb_run.finish()


if __name__ == "__main__":
    main()
