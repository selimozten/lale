# lale

Turkish instruct model distilled from frontier model outputs.

"Lale" means "tulip" in Turkish — symbol of the Ottoman golden age (Lale Devri).

## Why This Exists

Turkish speakers deserve a high-quality open instruct model. By distilling from Claude Opus 4.6 outputs and evaluating on terazi, we can build one with verifiable quality.

## Approach

### Data Generation (Distillation from Opus)

1. Generate diverse Turkish instruction-response pairs using Claude Opus 4.6 via AWS Bedrock
2. Categories of training data:
   - General conversation and Q&A in Turkish
   - Reasoning chains (math, logic, common sense) in Turkish
   - Tool use / function calling examples in Turkish
   - Financial analysis and reporting in Turkish
   - Legal document understanding in Turkish
   - Code generation with Turkish instructions
   - Translation tasks (TR<->EN, TR<->other)
3. Quality filtering: reject low-quality generations, deduplicate
4. Target: 50K-100K high-quality instruction pairs
5. $9,000 AWS credits shared with terazi project

### Base Model Selection

Primary candidates (evaluate both, pick winner):
- Llama 3.1 8B Instruct
- Qwen 2.5 7B Instruct

Selection criteria:
- Existing Turkish token coverage in tokenizer
- Baseline terazi scores before fine-tuning
- Community license terms

### Fine-Tuning with Unsloth

- Library: Unsloth for 2-4x faster QLoRA training
- Method: QLoRA (4-bit quantization + LoRA adapters)
- LoRA config: r=64, alpha=128, target all linear layers
- Training: 3-5 epochs, cosine schedule, bf16
- Hardware: single A100/H100 or multi-GPU consumer setup

### Evaluation

- Primary benchmark: terazi (our own benchmark)
- Compare against:
  - Base model (before fine-tune)
  - Existing Turkish models (if any)
  - Frontier models (Opus, GPT-4, Gemini) as upper bound
- Ablations: dataset size, LoRA rank, base model choice

## Output Artifacts

- HuggingFace model: `selimozten/lale-8b` (merged)
- HuggingFace adapter: `selimozten/lale-8b-lora` (LoRA weights only)
- Training dataset: `selimozten/lale-instruct-tr` (the distilled data)
- Training logs and eval results in repo

## Tech Stack

- Unsloth for fine-tuning
- PyTorch + transformers + peft
- AWS Bedrock SDK for data generation
- Weights & Biases for experiment tracking
- terazi for evaluation

## Project Structure

```
lale/
  README.md
  pyproject.toml
  lale/
    __init__.py
    generate/          # Instruction data generation from Opus
      generate_data.py
      templates/       # Prompt templates per category
      filters.py       # Quality filtering
    train/
      train.py         # Unsloth fine-tuning script
      config.yaml      # Training hyperparameters
    eval/
      run_terazi.py    # Evaluate on terazi benchmark
      compare.py       # Compare against baselines
  data/                # Generated training data (or HF download)
  results/             # Eval results, plots
  scripts/
    generate_data.sh
    train.sh
    eval.sh
```

## Milestones

1. Repo setup, project structure
2. Data generation pipeline: prompt templates + Opus API calls
3. Generate 50K Turkish instruction pairs, quality filter
4. Baseline eval: run terazi on candidate base models
5. Fine-tune with Unsloth, iterate on hyperparameters
6. Evaluate lale on terazi, compare against baselines
7. Publish model + adapter + dataset on HuggingFace
8. Write model card with full methodology and results

## Dependencies

- Requires terazi benchmark to be ready for evaluation (project 1)
- Shares AWS Bedrock credits with terazi

## Success Criteria

- Measurably better than base model on terazi benchmarks
- Competitive with frontier models on terazi-core (within 80% of Opus scores)
- Clean, reproducible training pipeline
- Published on HuggingFace with proper model card
