# lale

Turkish instruct model distilled from Claude Opus 4.6 outputs, fine-tuned with Unsloth.

"Lale" means "tulip" in Turkish -- symbol of the Ottoman golden age (Lale Devri).

## Approach

1. **Generate** high-quality Turkish instruction data using Claude Opus 4.6 via AWS Bedrock
2. **Filter** for quality, language, and deduplication
3. **Fine-tune** a small open model (Llama 3.1 8B or Qwen 2.5 7B) using Unsloth QLoRA
4. **Evaluate** on the [terazi](https://github.com/selimozten/terazi) benchmark

## Quick Start

```bash
pip install -e ".[dev]"

# Generate training data (requires AWS credentials with Bedrock access)
lale generate --category general --num-examples 10000

# Filter and prepare data
lale prepare --input data/raw/ --output data/filtered/

# Train (requires GPU)
lale train --config lale/train/config.yaml

# Evaluate on terazi
lale eval --model outputs/lale-8b --benchmark terazi
```

## Project Structure

```
lale/
  lale/
    generate/          # Data generation from Opus via Bedrock
      templates/       # Prompt templates per category
      filters.py       # Quality filtering and dedup
    train/             # Unsloth fine-tuning
    eval/              # Terazi evaluation and comparison
  data/                # Generated training data
  results/             # Evaluation results
  scripts/             # Shell scripts for full pipelines
```

## Data Categories

- General Turkish Q&A and conversation
- Reasoning (math, logic, common sense)
- Tool use / function calling
- Financial analysis and reporting
- Legal document understanding
- Code generation with Turkish instructions
- Translation (TR<->EN)

## Model Card

- **Base model:** Llama 3.1 8B Instruct (or Qwen 2.5 7B)
- **Method:** QLoRA (r=64, alpha=128) via Unsloth
- **Training data:** ~50K-100K Turkish instruction pairs distilled from Claude Opus 4.6
- **Benchmark:** [terazi](https://github.com/selimozten/terazi)
- **Results:** TBD

## Artifacts

- Model: [selimozten/lale-8b](https://huggingface.co/selimozten/lale-8b) (TBD)
- Adapter: [selimozten/lale-8b-lora](https://huggingface.co/selimozten/lale-8b-lora) (TBD)
- Dataset: [selimozten/lale-instruct-tr](https://huggingface.co/datasets/selimozten/lale-instruct-tr) (TBD)

## License

Apache 2.0
