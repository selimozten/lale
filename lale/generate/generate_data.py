"""Generate Turkish instruction data using Claude Opus via AWS Bedrock."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import requests as httpx
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

MODEL_ID = "us.anthropic.claude-opus-4-6-v1"
MAX_TOKENS = 8192
DEFAULT_REGION = "us-east-1"

# Rate limiting: max requests per minute to stay within Bedrock quotas
REQUESTS_PER_MINUTE = 30
_MIN_REQUEST_INTERVAL = 60.0 / REQUESTS_PER_MINUTE


class GeneratedExample(BaseModel):
    id: str
    category: str
    messages: list[dict[str, str]]


class GenerationConfig(BaseModel):
    category: str
    num_examples: int = 100
    batch_size: int = 5
    region: str = DEFAULT_REGION
    output_dir: Path = Path("data/raw")
    max_retries: int = 3
    retry_delay: float = 5.0
    temperature: float = 0.9
    api_key: str | None = None


def get_api_key(config: GenerationConfig) -> str:
    """Resolve the Bedrock API key from config or environment."""
    key = config.api_key or os.environ.get("BEDROCK_API_KEY")
    if not key:
        raise ValueError(
            "Bedrock API key required. Set BEDROCK_API_KEY env var "
            "or pass --api-key."
        )
    return key


def load_template(category: str) -> str:
    """Load the prompt template for a given category."""
    path = TEMPLATES_DIR / f"{category}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_text(encoding="utf-8")


def build_prompt(template: str, batch_size: int) -> str:
    """Build the full prompt for Opus to generate a batch of examples."""
    return (
        f"{template}\n\n"
        f"Simdi tam olarak {batch_size} adet ornek uret. "
        f"Her ornegi asagidaki JSON formatinda ver:\n\n"
        f'{{"messages": [{{"role": "user", "content": "..."}}, '
        f'{{"role": "assistant", "content": "..."}}]}}\n\n'
        f"Ornekleri bir JSON dizisi icinde ver. Sadece JSON dizisini dondur, "
        f"baska bir sey yazma. Ciktinin tamami gecerli JSON olmali."
    )


def call_bedrock(
    api_key: str,
    region: str,
    prompt: str,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    temperature: float = 0.9,
) -> str | None:
    """Call Claude Opus via Bedrock API key with retry logic."""
    url = (
        f"https://bedrock-runtime.{region}.amazonaws.com"
        f"/model/{MODEL_ID}/invoke"
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    })

    for attempt in range(max_retries):
        try:
            resp = httpx.post(url, headers=headers, data=payload, timeout=300)
            if resp.status_code == 429:
                if attempt < max_retries - 1:
                    wait = retry_delay * (2**attempt)
                    print(f"Throttled, waiting {wait:.0f}s...")
                    time.sleep(wait)
                    continue
                return None
            resp.raise_for_status()
            body = resp.json()
            return body["content"][0]["text"]
        except Exception as e:
            print(f"Error calling Bedrock (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return None
    return None


def parse_response(raw: str, category: str) -> list[GeneratedExample]:
    """Parse the model response into structured examples."""
    # Try to extract JSON array from the response
    text = raw.strip()
    # Handle cases where model wraps in markdown code block
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON array in the text
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
        # Validate message structure
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
    api_key = get_api_key(config)

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
          f"(have {existing}, target {config.num_examples})")

    num_batches = (remaining + config.batch_size - 1) // config.batch_size
    generated = 0

    last_request_time = 0.0

    with output_path.open("a", encoding="utf-8") as f:
        for _ in tqdm(range(num_batches), desc=config.category):
            if generated >= remaining:
                break

            # Rate limiting
            elapsed = time.monotonic() - last_request_time
            if elapsed < _MIN_REQUEST_INTERVAL:
                time.sleep(_MIN_REQUEST_INTERVAL - elapsed)

            prompt = build_prompt(template, config.batch_size)
            last_request_time = time.monotonic()
            raw = call_bedrock(
                api_key, config.region, prompt,
                max_retries=config.max_retries,
                retry_delay=config.retry_delay,
                temperature=config.temperature,
            )
            if raw is None:
                continue

            examples = parse_response(raw, config.category)
            for ex in examples:
                if generated >= remaining:
                    break
                f.write(ex.model_dump_json() + "\n")
                f.flush()
                generated += 1

    total = existing + generated
    failed = remaining - generated
    print(f"Done. {total} total examples for '{config.category}' in {output_path}")
    if failed > 0:
        print(f"Warning: {failed} examples could not be generated (API errors or parse failures)")
    return output_path
