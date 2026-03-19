#!/usr/bin/env python3
import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI
from config_loader import ApiConfig, load_api_config


STYLE_GUIDES = {
    "concise": "Write a short, keyword-like shopping query in 8-18 words.",
    "detailed": "Write a detailed shopping query in 1-2 sentences with concrete needs and constraints.",
    "feature_focused": "Write a query emphasizing functional features and compatibility details.",
    "scenario": "Write a query grounded in a realistic usage scenario and context.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate user queries from user histories with the last interaction as target item."
    )
    parser.add_argument(
        "--user-history-path",
        type=Path,
        default=Path("/home/threetu33/rec/output/user_histories.jsonl"),
        help="Path to user history JSONL (one user per line)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("/home/threetu33/rec/output/user_queries.jsonl"),
        help="Output JSONL path for generated query records",
    )
    parser.add_argument(
        "--max-history-items",
        type=int,
        default=20,
        help="Use at most this many most recent history interactions (excluding target)",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=10,
        help="Maximum number of users to process. 0 means all",
    )
    parser.add_argument(
        "--start-line",
        type=int,
        default=1,
        help="1-based start line in user history file",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional sleep between API calls",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per API call",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    return parser.parse_args()


def choose_style(user_id: str) -> str:
    styles = sorted(STYLE_GUIDES.keys())
    digest = hashlib.md5(user_id.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(styles)
    return styles[idx]


def safe_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").strip()
    if len(text) <= 240:
        return text
    return text[:237] + "..."


def format_history(history: List[Dict]) -> str:
    lines: List[str] = []
    for idx, item in enumerate(history, start=1):
        title = safe_text(item.get("title")) or "Unknown title"
        rating = item.get("rating")
        text = safe_text(item.get("text"))
        ts = item.get("timestamp")
        lines.append(
            f"{idx}. title={title} | rating={rating} | timestamp={ts} | review={text}"
        )
    return "\n".join(lines)


def build_prompt(style: str, user_id: str, history: List[Dict], target: Dict) -> str:
    target_title = safe_text(target.get("title")) or "Unknown title"
    target_rating = target.get("rating")
    target_text = safe_text(target.get("text"))
    style_hint = STYLE_GUIDES[style]

    return f"""
You are simulating a real user's shopping intent query for a recommendation experiment in musical instruments.

Task:
- Infer user preference from history.
- Generate ONE plausible pre-purchase user query that could lead to this target item.
- Follow the required style.

Style requirement:
{style_hint}

User ID:
{user_id}

User history (older interactions only):
{format_history(history)}

Intended target item (last interaction):
title={target_title}
rating={target_rating}
review={target_text}

Output rules:
- Return only the query text.
- Keep it natural and human-like.
- Do not mention internal fields like parent_asin/user_id/timestamp.
""".strip()


def call_model(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_retries: int,
) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You generate realistic e-commerce user search queries.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            content = (response.choices[0].message.content or "").strip()
            if content:
                return content
            raise RuntimeError("Empty response content")
        except Exception as err:
            last_err = err
            if attempt < max_retries:
                time.sleep(min(2.0 * attempt, 8.0))

    raise RuntimeError(f"Model call failed after retries: {last_err}")


def make_query_id(user_id: str, target: Dict) -> str:
    base = f"{user_id}|{target.get('parent_asin')}|{target.get('timestamp')}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def generate_queries(args: argparse.Namespace) -> None:
    if not args.user_history_path.exists():
        raise FileNotFoundError(f"User history file not found: {args.user_history_path}")
    api = load_api_config()
    model = api.default_model
    client = OpenAI(base_url=api.base_url, api_key=api.api_key, timeout=api.timeout_seconds)

    if args.max_history_items <= 0:
        raise ValueError("--max-history-items must be > 0")
    if args.start_line <= 0:
        raise ValueError("--start-line must be >= 1")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    errors = 0

    with args.user_history_path.open("r", encoding="utf-8") as src, args.output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_no, line in enumerate(src, start=1):
            if line_no < args.start_line:
                continue
            if args.max_users and processed >= args.max_users:
                break

            obj = json.loads(line)
            user_id = obj.get("user_id")
            interactions = obj.get("interactions", [])

            if not user_id or len(interactions) < 2:
                skipped += 1
                continue

            history = interactions[:-1]
            target = interactions[-1]

            history = history[-args.max_history_items :]
            if not history:
                skipped += 1
                continue

            style = choose_style(user_id)
            prompt = build_prompt(style=style, user_id=user_id, history=history, target=target)
            print(prompt)
            try:
                query = call_model(
                    client=client,
                    model=model,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                )
            except Exception as err:
                errors += 1
                result = {
                    "source_line": line_no,
                    "user_id": user_id,
                    "status": "error",
                    "error": str(err),
                }
                dst.write(json.dumps(result, ensure_ascii=False) + "\n")
                continue

            query_record = {
                "query_id": make_query_id(user_id, target),
                "source_line": line_no,
                "user_id": user_id,
                "query_style": style,
                "query_text": query,
                "target_item": {
                    "parent_asin": target.get("parent_asin"),
                    "title": target.get("title"),
                    "timestamp": target.get("timestamp"),
                    "rating": target.get("rating"),
                    "text": target.get("text"),
                },
                "history_items": history,
                "ua_query_packet": {
                    "query": query,
                    "style": style,
                    "user_id": user_id,
                    "target_parent_asin": target.get("parent_asin"),
                    "target_title": target.get("title"),
                    "target_timestamp": target.get("timestamp"),
                },
                "status": "ok",
            }
            dst.write(json.dumps(query_record, ensure_ascii=False) + "\n")
            processed += 1

            if processed % 50 == 0:
                print(f"processed={processed} skipped={skipped} errors={errors}")

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    print(
        json.dumps(
            {
                "output_path": str(args.output_path),
                "processed": processed,
                "skipped": skipped,
                "errors": errors,
                "start_line": args.start_line,
                "max_users": args.max_users,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    generate_queries(args)


if __name__ == "__main__":
    main()
