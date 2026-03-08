"""Generate Turkish instruction data using Claude via AWS Bedrock Converse API."""

from __future__ import annotations

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig
from pydantic import BaseModel
from tqdm import tqdm

CATEGORIES = [
    "general",
    "reasoning",
    "tool_use",
    "finance",
    "legal",
    "code",
    "translation",
]

TEMPLATES_DIR = Path(__file__).parent / "templates"

MODEL_ID = "us.anthropic.claude-sonnet-4-6"
MAX_TOKENS = 8192
DEFAULT_REGION = "us-east-1"
WORKERS = 20
BATCH_SIZE = 5

BOTO_CONFIG = BotoConfig(
    read_timeout=600,
    retries={"max_attempts": 3, "mode": "adaptive"},
)


class GeneratedExample(BaseModel):
    id: str
    category: str
    messages: list[dict[str, str]]


class GenerationConfig(BaseModel):
    category: str
    num_examples: int = 100
    batch_size: int = BATCH_SIZE
    workers: int = WORKERS
    region: str = DEFAULT_REGION
    output_dir: Path = Path("data/raw")
    temperature: float = 0.9
    api_key: str | None = None


def init_api_key(config: GenerationConfig) -> None:
    """Set the Bedrock API key in the environment for boto3."""
    key = config.api_key or os.environ.get("BEDROCK_API_KEY")
    if not key:
        raise ValueError(
            "Bedrock API key required. Set BEDROCK_API_KEY env var "
            "or pass --api-key."
        )
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = key


def load_template(category: str) -> str:
    """Load the prompt template for a given category."""
    path = TEMPLATES_DIR / f"{category}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_text(encoding="utf-8")


def build_prompt(template: str, batch_size: int) -> str:
    """Build the full prompt to generate a batch of examples."""
    return (
        f"{template}\n\n"
        f"Simdi tam olarak {batch_size} adet ornek uret. "
        f"Her ornegi asagidaki JSON formatinda ver:\n\n"
        f'{{"messages": [{{"role": "user", "content": "..."}}, '
        f'{{"role": "assistant", "content": "..."}}]}}\n\n'
        f"Ornekleri bir JSON dizisi icinde ver. Sadece JSON dizisini dondur, "
        f"baska bir sey yazma. Ciktinin tamami gecerli JSON olmali."
    )


def call_bedrock_once(
    region: str,
    prompt: str,
    temperature: float = 0.9,
) -> str | None:
    """Single Bedrock Converse call via boto3."""
    try:
        client = boto3.client("bedrock-runtime", region_name=region, config=BOTO_CONFIG)
        resp = client.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": temperature},
        )
        return resp["output"]["message"]["content"][0]["text"]
    except Exception as e:
        print(f"Bedrock error: {e}")
        return None


def parse_response(raw: str, category: str) -> list[GeneratedExample]:
    """Parse the model response into structured examples."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            try:
                items = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(items, list):
        items = [items]

    examples: list[GeneratedExample] = []
    for item in items:
        messages = item.get("messages", [])
        if not messages:
            continue
        if not all(
            isinstance(m, dict) and "role" in m and "content" in m for m in messages
        ):
            continue
        examples.append(
            GeneratedExample(
                id=str(uuid.uuid4()),
                category=category,
                messages=messages,
            )
        )
    return examples


def generate(config: GenerationConfig) -> Path:
    """Run the full data generation pipeline for a category."""
    template = load_template(config.category)
    init_api_key(config)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{config.category}.jsonl"

    # Count existing examples to support resumption
    existing = 0
    if output_path.exists():
        existing = sum(1 for _ in output_path.open())

    remaining = config.num_examples - existing
    if remaining <= 0:
        print(f"Already have {existing} examples for {config.category}, skipping.")
        return output_path

    print(f"Generating {remaining} examples for '{config.category}' "
          f"(have {existing}, target {config.num_examples}, "
          f"workers={config.workers}, batch={config.batch_size})")

    generated = 0

    with output_path.open("a", encoding="utf-8") as f:
        # Process in waves of concurrent requests
        pbar = tqdm(total=remaining, desc=config.category, unit="ex")
        while generated < remaining:
            # How many batches in this wave
            needed = remaining - generated
            num_batches = min(config.workers, (needed + config.batch_size - 1) // config.batch_size)

            prompt = build_prompt(template, config.batch_size)

            with ThreadPoolExecutor(max_workers=num_batches) as pool:
                futures = [
                    pool.submit(call_bedrock_once, config.region, prompt, config.temperature)
                    for _ in range(num_batches)
                ]
                for future in as_completed(futures):
                    raw = future.result()
                    if raw is None:
                        continue
                    examples = parse_response(raw, config.category)
                    for ex in examples:
                        if generated >= remaining:
                            break
                        f.write(ex.model_dump_json() + "\n")
                        generated += 1
                    f.flush()
                    pbar.update(len(examples))

        pbar.close()

    total = existing + generated
    print(f"Done. {total} total examples for '{config.category}' in {output_path}")
    return output_path
