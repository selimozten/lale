"""Fine-tune a base model on Turkish instruction data using Unsloth."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel


class LoraConfig(BaseModel):
    r: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: str = "all-linear"


class TrainingConfig(BaseModel):
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    lr_scheduler: str = "cosine"
    weight_decay: float = 0.01
    bf16: bool = True
    max_seq_length: int = 4096
    packing: bool = True


class DataConfig(BaseModel):
    train_path: str = "data/filtered/all.jsonl"
    eval_split: float = 0.02


class OutputConfig(BaseModel):
    dir: str = "outputs/lale-8b"
    save_adapter: bool = True
    save_merged: bool = False
    push_to_hub: bool = False
    hub_model_id: str = "selimozten/lale-8b"


class LoggingConfig(BaseModel):
    wandb_project: str = "lale"
    wandb_run_name: str | None = None
    log_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500


class Config(BaseModel):
    base_model: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    lora: LoraConfig = LoraConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    output: OutputConfig = OutputConfig()
    logging: LoggingConfig = LoggingConfig()


def load_config(path: Path) -> Config:
    """Load training config from YAML."""
    with path.open() as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


def load_dataset(config: Config) -> tuple:
    """Load and prepare the training dataset."""
    from datasets import Dataset

    data_path = Path(config.data.train_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {data_path}\n"
            f"Run 'lale generate' and 'lale prepare' first."
        )

    examples: list[dict] = []
    with data_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if not examples:
        raise ValueError(f"No examples found in {data_path}")

    print(f"Loaded {len(examples)} examples from {data_path}")

    ds = Dataset.from_list(examples)
    if config.data.eval_split > 0:
        split = ds.train_test_split(test_size=config.data.eval_split, seed=42)
        return split["train"], split["test"]
    return ds, None


def format_chat(example: dict, tokenizer) -> str:
    """Format a single example using the tokenizer's chat template."""
    messages = example["messages"]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def train(config_path: Path) -> None:
    """Run the full training pipeline."""
    config = load_config(config_path)

    # Lazy imports -- these are heavy
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    import wandb

    # Init wandb
    run_name = config.logging.wandb_run_name or f"lale-{config.base_model.split('/')[-1]}"
    wandb.init(project=config.logging.wandb_project, name=run_name)

    # Load model
    print(f"Loading base model: {config.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.training.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # Apply LoRA
    print(f"Applying LoRA: r={config.lora.r}, alpha={config.lora.alpha}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Load data
    print(f"Loading data from: {config.data.train_path}")
    train_ds, eval_ds = load_dataset(config)
    print(f"Train: {len(train_ds)} examples" + (f", Eval: {len(eval_ds)}" if eval_ds else ""))

    # Format examples
    def _format(example: dict) -> dict:
        return {"text": format_chat(example, tokenizer)}

    train_ds = train_ds.map(_format)
    if eval_ds:
        eval_ds = eval_ds.map(_format)

    # Training args
    output_dir = Path(config.output.dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler,
        weight_decay=config.training.weight_decay,
        bf16=config.training.bf16,
        logging_steps=config.logging.log_steps,
        save_steps=config.logging.save_steps,
        eval_steps=config.logging.eval_steps if eval_ds else None,
        eval_strategy="steps" if eval_ds else "no",
        report_to="wandb",
        save_total_limit=3,
        seed=42,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=config.training.max_seq_length,
        packing=config.training.packing,
    )

    # Resume from checkpoint if one exists
    last_checkpoint = None
    output_checkpoints = list(Path(config.output.dir).glob("checkpoint-*"))
    if output_checkpoints:
        last_checkpoint = str(sorted(output_checkpoints)[-1])
        print(f"Resuming from checkpoint: {last_checkpoint}")

    print("Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save
    if config.output.save_adapter:
        adapter_path = output_dir / "adapter"
        model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))
        print(f"Adapter saved to: {adapter_path}")

    if config.output.save_merged:
        merged_path = output_dir / "merged"
        model.save_pretrained_merged(str(merged_path), tokenizer, save_method="merged_16bit")
        print(f"Merged model saved to: {merged_path}")

    if config.output.push_to_hub:
        model.push_to_hub(config.output.hub_model_id)
        tokenizer.push_to_hub(config.output.hub_model_id)
        print(f"Pushed to hub: {config.output.hub_model_id}")

    wandb.finish()
    print("Training complete.")
