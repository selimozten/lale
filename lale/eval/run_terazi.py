"""Evaluate a model on the terazi benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class EvalConfig(BaseModel):
    model_path: str
    benchmark: str = "terazi"
    output_dir: Path = Path("results")
    max_samples: int | None = None


class EvalResult(BaseModel):
    model: str
    benchmark: str
    scores: dict[str, float]
    metadata: dict[str, str] = {}


def run_terazi(config: EvalConfig) -> EvalResult:
    """Run terazi benchmark on a model."""
    # Lazy imports
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from: {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        device_map="auto",
        torch_dtype="auto",
    )

    # Try to import and run terazi
    try:
        import terazi
        results = terazi.evaluate(
            model=model,
            tokenizer=tokenizer,
            max_samples=config.max_samples,
        )
        scores = results.scores
    except ImportError:
        print("terazi not installed. Install with: pip install terazi")
        print("Running placeholder evaluation...")
        scores = {"note": "terazi not installed -- placeholder results"}

    result = EvalResult(
        model=config.model_path,
        benchmark=config.benchmark,
        scores=scores,
        metadata={"model_path": config.model_path},
    )

    # Save results
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(config.model_path).name
    output_path = config.output_dir / f"{model_name}_{config.benchmark}.json"
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"Results saved to: {output_path}")

    return result


def print_results(result: EvalResult) -> None:
    """Print evaluation results as a table."""
    print(f"\n{'='*50}")
    print(f"Model: {result.model}")
    print(f"Benchmark: {result.benchmark}")
    print(f"{'='*50}")
    for key, value in result.scores.items():
        if isinstance(value, float):
            print(f"  {key:30s} {value:.4f}")
        else:
            print(f"  {key:30s} {value}")
    print(f"{'='*50}\n")
