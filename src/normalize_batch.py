"""
Async Normalization Script — description-only.
Auto-handles rate limits with server-suggested backoff. Resumable on crash.

Usage: python normalize_async.py
"""

import asyncio
import json
import re
import pandas as pd
from pathlib import Path
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APITimeoutError
from tqdm.asyncio import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "gpt-4o-mini"
MAX_TOKENS = 300
CONCURRENCY = 100

SPLITS = {
    "train":      "train_sample.csv",
    "validation": "validation_sample.csv",
    "test":       "test_sample.csv",
}

RESULTS_PATH = Path("results.jsonl")
OUTPUT_DIR = Path(".")

API_KEY = ""  # ← paste your key here

PROMPT = (
    "You are a text normalizer for a consumer marketplace price prediction system. "
    "You will receive a product description written by a seller. "
    "Your task is to rewrite it as clear, fluent prose. "
    "Expand all abbreviations and acronyms. Correct spelling errors. Normalize condition descriptions "
    "(e.g. 'gr8 cond' → 'great condition'). Improve sentence structure and clarity. "
    "Only interpret vague or non-standard terms when the meaning cannot be understood as written "
    "(e.g. 'skin energy drink' → 'facial moisturizer'). Do not add descriptive language, "
    "use cases, or editorial commentary that is not present in the original description. "
    "Remove self-promotional seller noise such as follower requests, discount offers for bundles, "
    "and unsolicited personal appeals — keep only information relevant to the product itself. "
    "IMPORTANT: Preserve all factual details exactly — brand names, model numbers, storage sizes, "
    "colors, and specifications must not be altered, invented, or removed. "
    "Do NOT introduce information that is not present in the original description. "
    "Do NOT infer the product's category, brand, size, or type if it is not stated in the description. "
    "Preserve the original sentiment and intensity of descriptive language — do not upgrade or "
    "downgrade adjectives (e.g. 'great' must not become 'fantastic', 'okay' must not become 'good'). "
    "Return only the rewritten description as plain prose, no bullet points or headers."
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_retry_after(error_message: str) -> float:
    """Extract suggested retry delay from a 429 error message. Returns seconds."""
    m_ms = re.search(r"try again in (\d+(?:\.\d+)?)ms", error_message)
    if m_ms:
        return max(0.3, float(m_ms.group(1)) / 1000.0)
    m_s = re.search(r"try again in (\d+(?:\.\d+)?)s", error_message)
    if m_s:
        return max(0.3, float(m_s.group(1)))
    return 2.0  # fallback


def load_completed_ids() -> set[str]:
    done = set()
    if not RESULTS_PATH.exists():
        return done
    with open(RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("text"):
                    done.add(rec["custom_id"])
            except json.JSONDecodeError:
                continue
    return done


def build_tasks() -> list[tuple[str, str]]:
    tasks = []
    for split, filename in SPLITS.items():
        df = pd.read_csv(filename)
        print(f"  Loaded {split}: {len(df)} rows")
        for _, row in df.iterrows():
            desc = row.get("item_description")
            if pd.isna(desc) or not str(desc).strip():
                continue
            cid = f"{split}_{row['train_id']}"
            tasks.append((cid, str(desc).strip()))
    return tasks


# ── Core ──────────────────────────────────────────────────────────────────────

async def normalize_one(client: AsyncOpenAI, sem: asyncio.Semaphore,
                        cid: str, text: str) -> tuple[str, str | None]:
    """Normalize one description. Retries 429s indefinitely with server-suggested delay.
    Retries connection/timeout errors with exponential backoff (5 attempts max).
    Other errors fail fast."""
    async with sem:
        connection_attempts = 0
        rate_limit_waits = 0
        while True:
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": PROMPT},
                        {"role": "user",   "content": text},
                    ],
                )
                return cid, resp.choices[0].message.content.strip()

            except RateLimitError as e:
                wait_s = parse_retry_after(str(e)) + 0.5
                rate_limit_waits += 1
                # Cap at 1000 rate-limit retries to avoid infinite loops in pathological cases
                if rate_limit_waits > 1000:
                    print(f"\n    Gave up on {cid} after 1000 rate-limit retries.")
                    return cid, None
                await asyncio.sleep(wait_s)

            except (APIConnectionError, APITimeoutError) as e:
                connection_attempts += 1
                if connection_attempts >= 5:
                    print(f"\n    Connection failed for {cid} after 5 attempts: {type(e).__name__}")
                    return cid, None
                await asyncio.sleep(2 ** connection_attempts)

            except Exception as e:
                # Unknown error — log and give up on this one
                print(f"\n    Unexpected error for {cid}: {type(e).__name__}: {e}")
                return cid, None


async def run_normalization(tasks: list[tuple[str, str]]):
    client = AsyncOpenAI(api_key=API_KEY, max_retries=0)  # we handle retries ourselves
    sem = asyncio.Semaphore(CONCURRENCY)

    f = open(RESULTS_PATH, "a", encoding="utf-8")
    try:
        coros = [normalize_one(client, sem, cid, text) for cid, text in tasks]
        for coro in tqdm.as_completed(coros, total=len(coros), desc="Normalizing"):
            cid, normalized = await coro
            f.write(json.dumps({"custom_id": cid, "text": normalized}, ensure_ascii=False) + "\n")
            f.flush()
    finally:
        f.close()
        await client.close()


def load_results() -> dict[str, str]:
    """Load streamed results. If a custom_id appears multiple times, last successful wins."""
    results = {}
    if not RESULTS_PATH.exists():
        return results
    with open(RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("text"):
                    results[rec["custom_id"]] = rec["text"]
            except json.JSONDecodeError:
                continue
    return results


def write_output_csvs(results: dict[str, str]):
    for split, filename in SPLITS.items():
        df = pd.read_csv(filename)
        out_rows = []
        missing = 0
        for _, row in df.iterrows():
            cid = f"{split}_{row['train_id']}"
            normalized = results.get(cid)
            if normalized is None:
                normalized = str(row.get("item_description", "")) if pd.notna(row.get("item_description")) else ""
                missing += 1
            out_rows.append({
                "train_id":         row["train_id"],
                "name":             row.get("name", ""),
                "category_name":    row.get("category_name", ""),
                "price":            row.get("price", ""),
                "item_description": normalized,
            })
        out_df = pd.DataFrame(out_rows)
        out_path = OUTPUT_DIR / f"{split}_p1.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  Wrote {len(out_df)} rows → {out_path}  (fallback to original: {missing})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Loading splits and building task list ===")
    all_tasks = build_tasks()
    print(f"  Total tasks: {len(all_tasks)}")

    completed = load_completed_ids()
    if completed:
        print(f"  Found {len(completed)} already-completed results — skipping.")
        all_tasks = [(cid, text) for cid, text in all_tasks if cid not in completed]
        print(f"  Remaining tasks: {len(all_tasks)}")

    if all_tasks:
        print(f"\n=== Running with concurrency={CONCURRENCY} ===")
        asyncio.run(run_normalization(all_tasks))
    else:
        print("  All tasks already completed.")

    print("\n=== Writing output CSVs ===")
    results = load_results()
    print(f"  Loaded {len(results)} successful results from {RESULTS_PATH}")
    write_output_csvs(results)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()