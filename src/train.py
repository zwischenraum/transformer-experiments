import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
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
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    dataset_state: dict,
    model_config: dict,
    total_tokens_seen: int,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "dataset": dataset_state,
            "model_config": model_config,
            "total_tokens_seen": total_tokens_seen,
        },
        path,
    )


def checkpoint_path_for_step(base_path: Path, step_value: int) -> Path:
    if step_value <= 0:
        raise ValueError("Step must be positive when building checkpoint path.")
    return base_path.with_name(f"{base_path.stem}-step-{step_value}{base_path.suffix}")


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
    checkpoint_interval_steps = int(training_config["checkpoint_interval_steps"])
    if checkpoint_interval_steps <= 0:
        raise ValueError("training.checkpoint_interval_steps must be positive.")
    checkpoint_base_path = Path(checkpoint_filename)

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
    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    warmup_steps = int(training_config["warmup_steps"])
    cosine_steps = total_steps - warmup_steps

    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-12, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=0.0)
    scheduler = SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    start_step = 0
    total_tokens_seen = 0
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
        scheduler.load_state_dict(checkpoint["scheduler"])
        dataset.load_state_dict(checkpoint["dataset"])
        start_step = int(checkpoint["step"])
        total_tokens_seen = int(checkpoint["total_tokens_seen"])

    def persist_checkpoint(
        *,
        current_step: int,
        target_path: Path,
        extra_aliases: list[str] | None = None,
    ) -> None:
        dataset_state = dataset.state_dict()
        save_checkpoint(
            path=target_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=current_step,
            dataset_state=dataset_state,
            model_config=model_config,
            total_tokens_seen=total_tokens_seen,
        )
        aliases = ["latest", f"step-{current_step}"]
        if extra_aliases:
            aliases.extend(extra_aliases)
        artifact = wandb.Artifact(artifact_name, type="model")
        artifact.add_file(str(target_path))
        artifact.metadata["step"] = current_step
        wandb_run.log_artifact(artifact, aliases=aliases)

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

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), torch.inf
        )  # no clipping yet - investigate the effects of clipping
        parameter_norm = torch.nn.utils.get_total_norm(model.parameters())

        optimizer.step()
        scheduler.step()

        tokens_in_batch = attention_mask.sum().item()
        total_tokens_seen += tokens_in_batch
        wandb_run.log(
            {
                "loss": loss.item(),
                "gradient_norm": gradient_norm.item(),
                "parameter_norm": parameter_norm.item(),
                "learning_rate": scheduler.get_last_lr()[0],
                "batch_tokens": tokens_in_batch,
                "total_tokens_seen": total_tokens_seen,
            },
            step=step,
        )
        pbar.set_description(f"Loss: {loss.item():.4f}")
        step += 1
        if step % checkpoint_interval_steps == 0:
            interval_checkpoint_path = checkpoint_path_for_step(
                checkpoint_base_path, step
            )
            persist_checkpoint(
                current_step=step,
                target_path=interval_checkpoint_path,
            )

    if step == start_step:
        raise RuntimeError(
            "No training steps were executed; increase total_steps to continue training."
        )

    final_checkpoint_path = checkpoint_base_path
    persist_checkpoint(
        current_step=step,
        target_path=final_checkpoint_path,
        extra_aliases=["final"],
    )
    wandb_run.finish()


if __name__ == "__main__":
    main()
