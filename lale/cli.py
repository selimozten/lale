"""CLI entry point for lale."""

from __future__ import annotations

from pathlib import Path

import click


@click.group()
@click.version_option(package_name="lale")
def cli() -> None:
    """lale -- Turkish instruct model distilled from frontier model outputs."""


@cli.command()
@click.option("--category", type=str, required=True, help="Data category to generate.")
@click.option("--num-examples", type=int, default=1000, help="Number of examples to generate.")
@click.option("--batch-size", type=int, default=5, help="Examples per API call.")
@click.option("--region", type=str, default="us-east-1", help="AWS region.")
@click.option("--output-dir", type=click.Path(), default="data/raw", help="Output directory.")
def generate(
    category: str,
    num_examples: int,
    batch_size: int,
    region: str,
    output_dir: str,
) -> None:
    """Generate Turkish instruction data using Claude Opus via Bedrock."""
    from lale.generate.generate_data import GenerationConfig, generate as run_generate, CATEGORIES

    if category == "all":
        categories = CATEGORIES
    else:
        categories = [category]

    for cat in categories:
        config = GenerationConfig(
            category=cat,
            num_examples=num_examples,
            batch_size=batch_size,
            region=region,
            output_dir=Path(output_dir),
        )
        run_generate(config)


@cli.command()
@click.option("--input", "input_dir", type=click.Path(exists=True), default="data/raw",
              help="Input directory with raw JSONL files.")
@click.option("--output", "output_dir", type=click.Path(), default="data/filtered",
              help="Output directory for filtered data.")
@click.option("--skip-language-check", is_flag=True, help="Skip language detection.")
def prepare(input_dir: str, output_dir: str, skip_language_check: bool) -> None:
    """Filter and prepare training data."""
    from lale.generate.filters import filter_dataset, FilterStats
    import json

    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    total_stats = FilterStats()
    all_output = out_path / "all.jsonl"

    # Process each JSONL file
    with all_output.open("w", encoding="utf-8") as combined:
        for jsonl_file in sorted(in_path.glob("*.jsonl")):
            filtered_path = out_path / jsonl_file.name
            stats = filter_dataset(jsonl_file, filtered_path, skip_language_check)
            click.echo(f"{jsonl_file.name}: {stats.passed}/{stats.total} passed")
            click.echo(f"  rejected: format={stats.rejected_format}, "
                       f"short={stats.rejected_short}, long={stats.rejected_long}, "
                       f"lang={stats.rejected_language}, dup={stats.rejected_duplicate}")

            # Append to combined file
            with filtered_path.open(encoding="utf-8") as f:
                for line in f:
                    combined.write(line)

            total_stats.total += stats.total
            total_stats.passed += stats.passed
            total_stats.rejected_short += stats.rejected_short
            total_stats.rejected_long += stats.rejected_long
            total_stats.rejected_language += stats.rejected_language
            total_stats.rejected_format += stats.rejected_format
            total_stats.rejected_duplicate += stats.rejected_duplicate

    click.echo(f"\nTotal: {total_stats.passed}/{total_stats.total} passed")
    click.echo(f"Combined output: {all_output}")


@cli.command()
@click.option("--config", "config_path", type=click.Path(exists=True),
              default="lale/train/config.yaml", help="Training config YAML.")
def train(config_path: str) -> None:
    """Fine-tune a model on the prepared data."""
    from lale.train.train import train as run_train
    run_train(Path(config_path))


@cli.command("eval")
@click.option("--model", type=str, required=True, help="Path to model.")
@click.option("--benchmark", type=str, default="terazi", help="Benchmark to run.")
@click.option("--output-dir", type=click.Path(), default="results", help="Results output dir.")
@click.option("--max-samples", type=int, default=None, help="Max samples to evaluate.")
def evaluate(model: str, benchmark: str, output_dir: str, max_samples: int | None) -> None:
    """Evaluate a model on a benchmark."""
    from lale.eval.run_terazi import EvalConfig, run_terazi, print_results

    config = EvalConfig(
        model_path=model,
        benchmark=benchmark,
        output_dir=Path(output_dir),
        max_samples=max_samples,
    )
    result = run_terazi(config)
    print_results(result)


@cli.command()
@click.option("--results-dir", type=click.Path(exists=True), default="results",
              help="Directory with result JSON files.")
@click.option("--output", type=click.Path(), default=None, help="Save table to file.")
def compare(results_dir: str, output: str | None) -> None:
    """Compare evaluation results across models."""
    from lale.eval.compare import compare as run_compare
    run_compare(
        Path(results_dir),
        Path(output) if output else None,
    )
