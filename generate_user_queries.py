#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

from config_loader import ApiConfig, load_api_config


STYLE_GUIDES = {
    "concise": "Write a short, keyword-like shopping query in 8-18 words with a direct purchase intent.",
    "scenario": "Write a query grounded in a realistic usage scenario and context.",
    "comparison": "Write a query that compares 2-3 alternatives or asks 'which is better' with criteria.",
    "compatibility": "Write a query emphasizing compatibility requirements (model, interface, size, format).",
    "beginner_friendly": "Write a query as a beginner asking for easy-to-use, low-risk recommendations.",
    "pro_grade": "Write a query as an experienced user emphasizing performance, durability, and pro features.",
    "upgrade": "Write a query from a user who already has some gear and now wants a clear upgrade or next-step purchase.",
    "troubleshooting": "Write a query driven by a concrete annoyance, limitation, or problem the user is trying to solve.",
    "taste_driven": "Write a query led by taste, vibe, sound character, aesthetics, or the kind of feeling the user wants.",
    "serendipitous": "Write a query that starts from an adjacent need, curiosity, or browsing interest and could plausibly lead the user to discover a different target item.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate user queries from user histories with the last interaction as target item."
    )
    parser.add_argument(
        "--user-history-path",
        type=Path,
        default=Path("./output/user_histories.jsonl"),
        help="Path to user history JSONL (one user per line)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("./output/user_queries.jsonl"),
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
        default=5,
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
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="Maximum number of concurrent query generations",
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


def parse_json_object(text: str) -> Optional[Dict]:
    body = (text or "").strip()
    if not body:
        return None

    try:
        obj = json.loads(body)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", body, flags=re.DOTALL)
    if not match:
        return None

    try:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def build_fallback_implicit_state(style: str, history: List[Dict], target: Dict) -> str:
    recent_titles = [
        safe_text(item.get("title"))
        for item in history[-2:]
        if safe_text(item.get("title"))
    ]
    target_title = safe_text(target.get("title"))
    style_signal = {
        "concise": "They are drawn to options that feel immediately right without wanting to over-explain why.",
        "scenario": "They picture how the item will fit into a lived routine and respond strongly to that imagined fit.",
        "comparison": "They compare options carefully on the surface, but the final pull is often a quieter emotional preference.",
        "compatibility": "They talk about practical fit, while privately caring whether the choice feels frictionless and reassuring long term.",
        "beginner_friendly": "They want to feel safe and competent, even if they cannot clearly state that emotional need.",
        "pro_grade": "They are quietly motivated by wanting their setup to feel more serious and identity-consistent.",
        "upgrade": "They are not only chasing utility; they want the next purchase to feel like a meaningful personal step forward.",
        "troubleshooting": "They describe the problem directly, but what really matters is removing a lingering low-level frustration or doubt.",
        "taste_driven": "They may justify the purchase practically, while actually responding most to whether it matches their taste and self-image.",
        "serendipitous": "They are open to being moved by something adjacent that unexpectedly feels more 'them' than the obvious match.",
    }.get(
        style,
        "They may say one thing explicitly while being guided by a softer, harder-to-name sense of personal fit.",
    )

    if recent_titles:
        history_signal = (
            "Their recent history suggests they keep circling back to "
            + ", ".join(recent_titles)
            + "."
        )
    elif target_title:
        history_signal = (
            f"They may be more responsive to something that resonates with the broader trajectory that eventually led to {target_title}."
        )
    else:
        history_signal = (
            "Their past choices suggest the final decision may depend on a subtle feeling of fit rather than explicit requirements alone."
        )

    return safe_text(f"{style_signal} {history_signal}")


def build_prompt(style: str, user_id: str, history: List[Dict], target: Dict) -> str:
    target_title = safe_text(target.get("title")) or "Unknown title"
    target_rating = target.get("rating")
    target_text = safe_text(target.get("text"))
    style_hint = STYLE_GUIDES[style]

    return f"""
You are simulating a real user's shopping intent query for a recommendation experiment in musical instruments.

Task:
- Infer user preference from history.
- Generate ONE plausible pre-purchase user query that could plausibly lead into the broader shopping path.
- Also generate ONE hidden implicit user state that the user would not clearly articulate and would not tell a user agent.
- Follow the required style.

Style requirement:
{style_hint}

User history (older interactions only):
{format_history(history)}

Intended target item (last interaction):
title={target_title}
rating={target_rating}
review={target_text}

Output rules:
- Return JSON only with exactly these keys:
  {{
    "query_text": "...",
    "implicit_state": "..."
  }}
- query_text:
  - natural and human-like
  - one shopping query only
  - must follow the requested style
  - should sound like what the user would actually type or say before purchase
- implicit_state:
  - 1 to 2 sentences
  - describe a subtle inner state, psychological pull, taste bias, or unspoken tendency
  - it should come from the user's history + style + target item together
  - it may be lightly constrained by the target item, but must NOT simply restate the target title or explicit query need
  - it should help explain why the final user choice may differ from the most obviously query-relevant item
  - it should feel real-world and somewhat hard for the user to verbalize
- Do not mention internal fields like parent_asin/user_id/timestamp.
""".strip()


def parse_generation_output(raw_text: str, style: str, history: List[Dict], target: Dict) -> Dict[str, str]:
    obj = parse_json_object(raw_text)
    if obj:
        query_text = safe_text(obj.get("query_text")) or safe_text(raw_text)
        implicit_state = safe_text(obj.get("implicit_state"))
        if query_text and implicit_state:
            return {
                "query_text": query_text,
                "implicit_state": implicit_state,
            }

    query_text = safe_text(raw_text)
    if not query_text:
        query_text = "looking for a good option that feels right for what I need"

    return {
        "query_text": query_text,
        "implicit_state": build_fallback_implicit_state(style, history, target),
    }


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
                        "content": (
                            "You generate realistic e-commerce user search intent records in strict JSON. "
                            "Never return markdown."
                        ),
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


def make_client(api: ApiConfig) -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("Missing dependency: openai. Install with `pip install openai`.")
    return OpenAI(base_url=api.base_url, api_key=api.api_key, timeout=api.timeout_seconds)


def build_query_job(line_no: int, obj: Dict, max_history_items: int) -> Optional[Dict]:
    user_id = obj.get("user_id")
    interactions = obj.get("interactions", [])

    if not user_id or len(interactions) < 2:
        return None

    history = interactions[:-1][-max_history_items:]
    if not history:
        return None

    target = interactions[-1]
    style = choose_style(user_id)
    return {
        "source_line": line_no,
        "user_id": user_id,
        "history": history,
        "target": target,
        "style": style,
    }


def generate_query_record(
    job: Dict,
    api: ApiConfig,
    model: str,
    temperature: float,
    max_retries: int,
    sleep_seconds: float,
) -> Dict:
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    user_id = job["user_id"]
    history = job["history"]
    target = job["target"]
    style = job["style"]
    prompt = build_prompt(style=style, user_id=user_id, history=history, target=target)

    try:
        raw_text = call_model(
            client=make_client(api),
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_retries=max_retries,
        )
    except Exception as err:
        return {
            "source_line": job["source_line"],
            "user_id": user_id,
            "status": "error",
            "error": str(err),
        }

    generated = parse_generation_output(
        raw_text=raw_text,
        style=style,
        history=history,
        target=target,
    )

    return {
        "query_id": make_query_id(user_id, target),
        "source_line": job["source_line"],
        "user_id": user_id,
        "user_context": {
            "query_text": generated["query_text"],
            "persona_style": style,
            "implicit_state": generated["implicit_state"],
        },
        "target_item": {
            "parent_asin": target.get("parent_asin"),
            "title": target.get("title"),
            "timestamp": target.get("timestamp"),
            "rating": target.get("rating"),
            "text": target.get("text"),
        },
        "history_items": history,
        "status": "ok",
    }


def generate_queries(args: argparse.Namespace) -> None:
    if not args.user_history_path.exists():
        raise FileNotFoundError(f"User history file not found: {args.user_history_path}")
    api = load_api_config()
    model = api.default_model

    if args.max_history_items <= 0:
        raise ValueError("--max-history-items must be > 0")
    if args.start_line <= 0:
        raise ValueError("--start-line must be >= 1")
    if args.max_concurrency <= 0:
        raise ValueError("--max-concurrency must be > 0")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    errors = 0

    # Best-effort total for progress display (counts raw lines, not necessarily valid users).
    with args.user_history_path.open("r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    remaining_lines = max(0, total_lines - args.start_line + 1)
    total_target = remaining_lines if args.max_users == 0 else min(remaining_lines, args.max_users)

    with args.user_history_path.open("r", encoding="utf-8") as src, args.output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        worker = partial(
            generate_query_record,
            api=api,
            model=model,
            temperature=args.temperature,
            max_retries=args.max_retries,
            sleep_seconds=args.sleep_seconds,
        )

        line_no = 0
        with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
            while True:
                remaining_quota = args.max_concurrency
                if args.max_users:
                    remaining_quota = min(remaining_quota, args.max_users - processed)
                    if remaining_quota <= 0:
                        break

                batch_jobs: List[Dict] = []
                while len(batch_jobs) < remaining_quota:
                    line = src.readline()
                    if not line:
                        break
                    line_no += 1
                    if line_no < args.start_line:
                        continue

                    job = build_query_job(
                        line_no=line_no,
                        obj=json.loads(line),
                        max_history_items=args.max_history_items,
                    )
                    if job is None:
                        skipped += 1
                        continue
                    batch_jobs.append(job)

                if not batch_jobs:
                    break

                for result in executor.map(worker, batch_jobs):
                    dst.write(json.dumps(result, ensure_ascii=False) + "\n")
                    if result.get("status") == "ok":
                        processed += 1
                    else:
                        errors += 1

                    sys.stderr.write(
                        f"\r[generate_user_queries] {processed}/{total_target} processed | skipped={skipped} errors={errors}"
                    )
                    sys.stderr.flush()

    sys.stderr.write("\n")
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
