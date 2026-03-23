#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from openai import OpenAI
from config_loader import ApiConfig, load_api_config


PITCH_STYLE_GUIDES = {
    "honest": "Use a transparent, factual tone. Mention trade-offs when relevant and avoid hype.",
    "expert": "Use a calm, expert tone. Highlight the most decision-critical specs and who it's for.",
    "concise": "Be extremely concise: one sentence, no fluff, focus on the single strongest reason.",
    "friendly": "Use a warm, friendly tone. Make it easy to understand and avoid jargon.",
    "promo": "Use a promotional, high-energy tone while staying plausible and not inventing specs.",
}


@dataclass
class Item:
    parent_asin: str
    title: str
    main_category: str
    average_rating: Optional[float]
    rating_number: Optional[int]
    description: str
    reviews: List[str]

    def to_pitch_payload(self) -> Dict:
        return {
            "parent_asin": self.parent_asin,
            "title": self.title,
            "main_category": self.main_category,
            "average_rating": self.average_rating,
            "rating_number": self.rating_number,
            "description": self.description,
            "review_snippets": self.reviews,
        }


@dataclass
class PlatformCandidate:
    platform_id: str
    item: Item
    pitch: str
    pitch_style: str
    retrieval_score: float
    forced_intended_hit: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run UA-PA simulation: structured query, platform recall, UA ranking, settlement."
    )
    parser.add_argument(
        "--user-queries-path",
        type=Path,
        default=Path("/home/threetu33/rec/output/user_queries.jsonl"),
        help="Path to generated user query JSONL",
    )
    parser.add_argument(
        "--platform-dir",
        type=Path,
        default=Path("/home/threetu33/rec/output/platforms"),
        help="Directory containing platform_1.jsonl ... platform_5.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/threetu33/rec/output/sim"),
        help="Output directory for simulation artifacts",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=5,
        help="Maximum number of queries to run. 0 means all",
    )
    parser.add_argument(
        "--start-line",
        type=int,
        default=1,
        help="1-based starting line in user query JSONL",
    )
    parser.add_argument(
        "--intended-hit-prob",
        type=float,
        default=0.9,
        help="If intended item exists in platform catalog, chance to force-hit it",
    )
    parser.add_argument(
        "--profile-window-size",
        type=int,
        default=5,
        help="Sliding window size for platform reputation memory entries",
    )
    parser.add_argument(
        "--max-platform-items",
        type=int,
        default=4000,
        help="Max items loaded per platform; 0 means load all",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="HuggingFace SentenceTransformer model name for dense embeddings",
    )
    parser.add_argument(
        "--embedding-local-only",
        action="store_true",
        help="Only load embedding model from local files (no network)",
    )
    parser.add_argument(
        "--embedding-device",
        type=str,
        default="cpu",
        help="Device for embedding model: cpu or cuda",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Batch size for embedding encoding",
    )
    parser.add_argument(
        "--embedding-max-seq-length",
        type=int,
        default=256,
        help="Max sequence length for embedding model",
    )
    parser.add_argument(
        "--embedding-show-progress",
        action="store_true",
        help="Show embedding encode progress bar",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries for each API request",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional sleep between query rounds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="Maximum number of queries to prepare concurrently",
    )
    return parser.parse_args()


def make_client(api: ApiConfig) -> OpenAI:
    return OpenAI(base_url=api.base_url, api_key=api.api_key, timeout=api.timeout_seconds)


def make_stable_rng(seed: int, *parts: object) -> random.Random:
    key = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(f"{seed}|{key}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def clean_text(text: Optional[str], limit: int = 600) -> str:
    if text is None:
        return ""
    t = str(text).replace("\n", " ").replace("\r", " ").strip()
    if len(t) <= limit:
        return t
    return t[: limit - 3] + "..."


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def hash_embedding(text: str, dim: int) -> List[float]:
    vec = [0.0] * dim
    tokens = tokenize(text)
    if not tokens:
        return vec

    for tok in tokens:
        digest = hashlib.md5(tok.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % dim
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vec[idx] += sign

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    # Deprecated: kept for backward compatibility with earlier experiments.
    return sum(x * y for x, y in zip(a, b))


def llm_chat_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int,
) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            content = (resp.choices[0].message.content or "").strip()
            if content:
                return content
            raise RuntimeError("Empty LLM response")
        except Exception as err:
            last_err = err
            if attempt < max_retries:
                time.sleep(min(2.0 * attempt, 8.0))
    raise RuntimeError(f"LLM request failed after retries: {last_err}")


def parse_json_object(text: str) -> Optional[Dict]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def parse_rank_list(text: str, valid_ids: List[str]) -> Optional[List[str]]:
    obj = parse_json_object(text)
    if obj:
        raw = obj.get("ranked_platform_ids") or obj.get("rank_list")
        if isinstance(raw, list):
            rank = [str(x) for x in raw]
            if sorted(rank) == sorted(valid_ids):
                return rank

    ids = re.findall(r"P\d+", text)
    if not ids:
        return None

    rank: List[str] = []
    seen = set()
    for pid in ids:
        if pid in valid_ids and pid not in seen:
            rank.append(pid)
            seen.add(pid)
    if sorted(rank) == sorted(valid_ids):
        return rank
    return None


def choose_pitch_style(platform_id: str) -> str:
    platform_style_map = {
        "P1": "honest",
        "P2": "expert",
        "P3": "concise",
        "P4": "friendly",
        "P5": "promo",
    }
    style = platform_style_map.get(platform_id)
    if style in PITCH_STYLE_GUIDES:
        return style
    return "honest"


def build_item_from_obj(obj: Dict) -> Item:
    desc = obj.get("description")
    if isinstance(desc, list):
        desc_text = clean_text(" ".join(str(x) for x in desc if x is not None), 1200)
    else:
        desc_text = clean_text(str(desc or ""), 1200)

    reviews = obj.get("reviews") or []
    snippets: List[str] = []
    for r in reviews:
        txt = clean_text((r or {}).get("text"), 220)
        if txt:
            snippets.append(txt)
        if len(snippets) >= 2:
            break

    return Item(
        parent_asin=str(obj.get("parent_asin") or ""),
        title=clean_text(obj.get("title"), 300),
        main_category=clean_text(obj.get("main_category"), 120),
        average_rating=obj.get("average_rating"),
        rating_number=obj.get("rating_number"),
        description=desc_text,
        reviews=snippets,
    )


def item_retrieval_text(item: Item) -> str:
    parts = [
        item.title,
        item.main_category,
        item.description,
        " ".join(item.reviews),
    ]
    return clean_text(" ".join(p for p in parts if p), 2200)


def load_platform_items(
    platform_file: Path,
    max_items: int,
    embedder,
    embedding_batch_size: int,
    embedding_show_progress: bool,
) -> Tuple[List[Item], Dict[str, Item], "np.ndarray"]:
    items: List[Item] = []
    asin_map: Dict[str, Item] = {}
    doc_texts: List[str] = []

    t_read0 = time.time()
    with platform_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if max_items and len(items) >= max_items:
                break
            obj = json.loads(line)
            item = build_item_from_obj(obj)
            if not item.parent_asin:
                continue
            text = item_retrieval_text(item)
            items.append(item)
            asin_map[item.parent_asin] = item
            doc_texts.append(text)

    # Dense embeddings (normalized) for cosine via dot product.
    import numpy as np

    if not doc_texts:
        vectors = np.zeros((0, 1), dtype=np.float32)
    else:
        print(f"  embedding_start docs={len(doc_texts)} batch={embedding_batch_size}")
        t0 = time.time()
        vectors = embedder.encode(
            doc_texts,
            batch_size=max(1, int(embedding_batch_size)),
            normalize_embeddings=True,
            show_progress_bar=bool(embedding_show_progress),
        ).astype(np.float32, copy=False)
        dt = time.time() - t0
        if dt > 0:
            print(f"  embedding_done docs={len(doc_texts)} dim={vectors.shape[1]} time_s={dt:.1f} docs_per_s={len(doc_texts)/dt:.1f}")
    return items, asin_map, vectors


def ua_structure_query(
    client: OpenAI,
    model: str,
    max_retries: int,
    query_record: Dict,
) -> Dict:
    query = query_record.get("query_text", "")
    style = query_record.get("query_style", "")

    user_prompt = f"""
Convert the following user query into a structured JSON object for downstream retrieval.
Output JSON only (no markdown, no extra keys).

Schema (types + limits):
- user_need: string (<= 25 words)
- query_rewrite: string (<= 25 words). Keep meaning; do NOT add new brands/models/specs not implied by the query.
- keywords: array of 3-8 short keyword phrases (nouns or noun-phrases). Avoid adjectives-only.
- constraints: array of 0-5 short constraints .

Example output:
{{
  "user_need": "A beginner-friendly electric guitar bundle for home practice",
  "query_rewrite": "beginner electric guitar starter kit for home practice",
  "keywords": ["electric guitar", "starter kit", "beginner", "home practice"],
  "constraints": ["easy to set up", "includes essentials", "quiet practice option"]
}}

Input:
query_text={query}
""".strip()

    system = "You are a strict JSON generator for recommendation query understanding."

    try:
        raw = llm_chat_json(client, model, system, user_prompt, max_retries)
        obj = parse_json_object(raw)
        if obj:
            obj.setdefault("style", style)
            obj.setdefault("query_rewrite", query)
            obj.setdefault("keywords", tokenize(query)[:6])
            obj.setdefault("constraints", [])
            obj.setdefault("user_need", query)
            return obj
    except Exception:
        pass

    return {
        "user_need": query,
        "query_rewrite": query,
        "keywords": tokenize(query)[:6],
        "constraints": [],
        "style": style,
    }


def build_query_embedding_text(ua_structured: Dict) -> str:
    return " ".join(
        [
            str(ua_structured.get("query_rewrite") or ""),
            " ".join(str(x) for x in (ua_structured.get("keywords") or [])),
            " ".join(str(x) for x in (ua_structured.get("constraints") or [])),
            str(ua_structured.get("user_need") or ""),
        ]
    )


def encode_query_vector(ua_structured: Dict, embedder, embedder_lock: threading.Lock):
    import numpy as np

    with embedder_lock:
        qvec = embedder.encode(
            [build_query_embedding_text(ua_structured)],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    return np.asarray(qvec, dtype=np.float32).reshape(-1)


def retrieve_top1(
    items: List[Item],
    vectors,
    qvec,
) -> Tuple[Item, float]:
    if not items:
        raise ValueError("Platform has no items loaded")
    if vectors is None or getattr(vectors, "shape", (0,))[0] == 0:
        raise ValueError("Platform has no embedding vectors loaded")

    scores = vectors @ qvec
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])
    return items[best_idx], best_score


def generate_platform_pitch(
    client: OpenAI,
    model: str,
    max_retries: int,
    platform_id: str,
    ua_structured: Dict,
    item: Item,
    pitch_style: str,
) -> str:
    style_instruction = PITCH_STYLE_GUIDES.get(
        pitch_style,
        "No extra style constraints. Write naturally.",
    )

    prompt = f"""
You are the bidding agent for platform {platform_id}. Write a concise product pitch
for the candidate item based on the user's structured needs.

Pitch style: {pitch_style}
Style guidance: {style_instruction}

Requirements:
- 1 to 3 sentences
- Do not invent specs or guarantees that are not in the provided item data
- Tie the message to user needs and usage scenario
- Return plain text only

User need structure:
{json.dumps(ua_structured, ensure_ascii=False)}

Candidate item data:
{json.dumps(item.to_pitch_payload(), ensure_ascii=False)}
""".strip()

    try:
        return llm_chat_json(
            client=client,
            model=model,
            system_prompt="You write concise product recommendation pitches.",
            user_prompt=prompt,
            max_retries=max_retries,
        )
    except Exception:
        pass

    return f"{item.title}. {item.description[:160]}"


def serialize_candidate(candidate: PlatformCandidate) -> Dict:
    return {
        "platform_id": candidate.platform_id,
        "item": candidate.item.to_pitch_payload(),
        "pitch": candidate.pitch,
        "pitch_style": candidate.pitch_style,
        "retrieval_score": candidate.retrieval_score,
        "forced_intended_hit": candidate.forced_intended_hit,
    }


def prepare_round(
    qrec: Dict,
    line_no: int,
    api: ApiConfig,
    chat_model: str,
    max_retries: int,
    sleep_seconds: float,
    seed: int,
    intended_hit_prob: float,
    platform_ids: List[str],
    platform_data: Dict[str, Dict],
    embedder,
    embedder_lock: threading.Lock,
) -> Dict:
    user_query = str(qrec.get("query_text") or "")
    target_item = qrec.get("target_item") or {}
    target_asin = str(target_item.get("parent_asin") or "")
    query_id = str(qrec.get("query_id") or f"line_{line_no}")
    round_obj = {
        "query_id": query_id,
        "source_line": line_no,
        "user_id": qrec.get("user_id"),
        "target_asin": target_asin,
        "status": "started",
    }

    try:
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        client = make_client(api)
        ua_struct = ua_structure_query(
            client=client,
            model=chat_model,
            max_retries=max_retries,
            query_record=qrec,
        )
        round_obj["ua_structured_query"] = ua_struct

        hit_rng = make_stable_rng(seed, query_id, "intended_hit")
        intended_hit_pids: List[str] = []
        intended_exists_any = False
        for pid in platform_ids:
            asin_map: Dict[str, Item] = platform_data[pid]["asin_map"]
            if target_asin in asin_map:
                intended_exists_any = True
                if hit_rng.random() < intended_hit_prob:
                    intended_hit_pids.append(pid)

        if not intended_exists_any:
            round_obj.update(
                {
                    "status": "skipped_no_intended",
                    "intended_hit_platform_ids": [],
                }
            )
            return {"round_obj": round_obj}

        if not intended_hit_pids:
            round_obj.update(
                {
                    "status": "skipped_intended_not_hit",
                    "intended_hit_platform_ids": [],
                }
            )
            return {"round_obj": round_obj}

        qvec = encode_query_vector(ua_struct, embedder, embedder_lock)
        candidates: List[PlatformCandidate] = []
        for pid in platform_ids:
            pdata = platform_data[pid]
            asin_map: Dict[str, Item] = pdata["asin_map"]
            items: List[Item] = pdata["items"]
            vectors = pdata["vectors"]

            forced_hit = False
            if pid in intended_hit_pids:
                chosen = asin_map[target_asin]
                score = 1.0
                forced_hit = True
            else:
                chosen, score = retrieve_top1(items=items, vectors=vectors, qvec=qvec)

            style = choose_pitch_style(pid)
            pitch = generate_platform_pitch(
                client=client,
                model=chat_model,
                max_retries=max_retries,
                platform_id=pid,
                ua_structured=ua_struct,
                item=chosen,
                pitch_style=style,
            )
            candidates.append(
                PlatformCandidate(
                    platform_id=pid,
                    item=chosen,
                    pitch=pitch,
                    pitch_style=style,
                    retrieval_score=score,
                    forced_intended_hit=forced_hit,
                )
            )

        intended_platforms = [
            candidate.platform_id
            for candidate in candidates
            if candidate.item.parent_asin == target_asin
        ]
        if not intended_platforms:
            round_obj.update(
                {
                    "status": "skipped_no_intended",
                    "platform_candidates": [serialize_candidate(c) for c in candidates],
                }
            )
            return {"round_obj": round_obj}

        return {
            "round_obj": round_obj,
            "user_query": user_query,
            "target_asin": target_asin,
            "candidates": candidates,
            "intended_platform_ids": intended_platforms,
        }
    except Exception as err:
        round_obj.update({"status": "error", "error": str(err)})
        return {"round_obj": round_obj}


def ua_rank_candidates(
    client: OpenAI,
    model: str,
    max_retries: int,
    user_query: str,
    candidates: List[PlatformCandidate],
    platform_profiles: Dict[str, List[str]],
    rng: random.Random,
) -> Tuple[List[str], str]:
    shuffled = candidates[:]
    rng.shuffle(shuffled)

    blocks: List[str] = []
    valid_ids = [c.platform_id for c in shuffled]
    for c in shuffled:
        profile_entries = platform_profiles.get(c.platform_id, [])
        profile_text = " | ".join(profile_entries) if profile_entries else ""
        block = {
            "platform_id": c.platform_id,
            "history_reputation_profile": profile_text,
            "candidate_item": c.item.to_pitch_payload(),
            "bidding_pitch": c.pitch,
        }
        blocks.append(json.dumps(block, ensure_ascii=False))

    prompt = f"""
You are a User Agent that prioritizes user benefit. Rank platform candidates by how
well they satisfy user needs.
You must consider historical reputation profiles. If a platform has records of
over-marketing or mismatch, lower its rank.

User need: {user_query}

Candidate list (already shuffled):
{chr(10).join(blocks)}

Output JSON only:
{{
  "ranked_platform_ids": ["P1","P2","P3","P4","P5"],
    "rationale": "brief explanation"
}}
ranked_platform_ids must include all platforms exactly once.
""".strip()

    raw = llm_chat_json(
        client=client,
        model=model,
        system_prompt="You are a strict ranking judge for recommendation platforms.",
        user_prompt=prompt,
        max_retries=max_retries,
    )

    rank = parse_rank_list(raw, valid_ids)
    if rank is None:
        # Fallback: keep shuffled order when parser fails
        rank = valid_ids
        rationale = "fallback_order_due_to_parse_failure"
    else:
        obj = parse_json_object(raw) or {}
        rationale = str(obj.get("rationale") or "")

    return rank, rationale


def generate_reputation_memory(
    client: OpenAI,
    model: str,
    max_retries: int,
    event_type: str,
    platform_id: str,
    user_query: str,
    item_title: str,
    rank_pos: int,
    target_rank: int,
) -> str:
    prompt = f"""
You are a platform reputation logger. Write ONE short English memory note (one sentence, <= 28 words).
It should be specific enough to help future ranking, but avoid claiming certainty beyond the evidence.

event_type={event_type}
platform_id={platform_id}
user_query={user_query}
recommended_item_title={item_title}
rank_pos={rank_pos}
target_rank={target_rank}

Rules:
- penalty: mention it was ranked ahead of the target but missed the intended match; add ONE plausible mismatch hint (e.g., wrong type, missing compatibility, budget mismatch), phrased as "may".
- reward: mention it matched the intended need; add ONE concrete helpful aspect aligned with the query (e.g., compatibility, use-case fit), without inventing specs.
""".strip()
    try:
        return llm_chat_json(
            client=client,
            model=model,
            system_prompt="You create concise reputation memory notes.",
            user_prompt=prompt,
            max_retries=max_retries,
        )
    except Exception:
        if event_type == "penalty":
            return "Ranked high but missed the true target, showing possible over-marketing risk."
        return "Matched the true need; recommendation appears trustworthy."


def append_profile_memory(
    profiles: Dict[str, List[str]],
    platform_id: str,
    entry: str,
    window_size: int,
) -> None:
    if platform_id not in profiles:
        profiles[platform_id] = []
    profiles[platform_id].append(entry)
    if len(profiles[platform_id]) > window_size:
        profiles[platform_id] = profiles[platform_id][-window_size:]


def load_platform_profiles(path: Path, platform_ids: List[str]) -> Dict[str, List[str]]:
    if path.exists():
        obj = json.loads(path.read_text(encoding="utf-8"))
        result: Dict[str, List[str]] = {}
        for pid in platform_ids:
            val = obj.get(pid, [])
            result[pid] = [str(x) for x in val] if isinstance(val, list) else []
        return result

    return {pid: [] for pid in platform_ids}


def persist_platform_profiles(path: Path, profiles: Dict[str, List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")


def run_simulation(args: argparse.Namespace) -> None:
    if args.start_line <= 0:
        raise ValueError("--start-line must be >= 1")
    if args.max_queries < 0:
        raise ValueError("--max-queries must be >= 0")
    if not (0.0 <= args.intended_hit_prob <= 1.0):
        raise ValueError("--intended-hit-prob must be in [0, 1]")
    if args.profile_window_size <= 0:
        raise ValueError("--profile-window-size must be > 0")
    if args.embedding_batch_size <= 0:
        raise ValueError("--embedding-batch-size must be > 0")
    if args.max_concurrency <= 0:
        raise ValueError("--max-concurrency must be > 0")

    platform_files = sorted(args.platform_dir.glob("platform_*.jsonl"))
    if len(platform_files) != 5:
        raise FileNotFoundError("Expected exactly 5 platform files named platform_1.jsonl..platform_5.jsonl")

    platform_ids = [f"P{i+1}" for i in range(len(platform_files))]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rounds_path = output_dir / "simulation_rounds.jsonl"
    profiles_path = output_dir / "platform_profiles.json"
    summary_path = output_dir / "simulation_summary.json"

    api = load_api_config()
    client = make_client(api)
    chat_model = api.default_model

    print(f"Loading embedding model: {args.embedding_model} (device={args.embedding_device})")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as err:
        raise RuntimeError(
            "Missing dependency: sentence-transformers. Install with `pip install -U sentence-transformers`."
        ) from err
    try:
        embedder = SentenceTransformer(
            args.embedding_model,
            device=args.embedding_device,
            local_files_only=bool(args.embedding_local_only),
        )
    except TypeError:
        # Older sentence-transformers versions may not support local_files_only.
        if args.embedding_local_only:
            raise RuntimeError(
                "Your sentence-transformers version doesn't support local_files_only. "
                "Please upgrade with `pip install -U sentence-transformers`."
            )
        embedder = SentenceTransformer(args.embedding_model, device=args.embedding_device)
    if args.embedding_max_seq_length > 0:
        embedder.max_seq_length = int(args.embedding_max_seq_length)
        print(f"Embedding max_seq_length set to {embedder.max_seq_length}")

    print("Loading platform catalogs and building retrieval vectors...")
    platform_data = {}
    for idx, pfile in enumerate(platform_files):
        pid = platform_ids[idx]
        print(f"{pid}: reading_items+embedding from {pfile} ...")
        items, asin_map, vectors = load_platform_items(
            platform_file=pfile,
            max_items=args.max_platform_items,
            embedder=embedder,
            embedding_batch_size=args.embedding_batch_size,
            embedding_show_progress=args.embedding_show_progress,
        )
        if not items:
            raise RuntimeError(f"{pid} has no loaded items: {pfile}")
        platform_data[pid] = {
            "items": items,
            "asin_map": asin_map,
            "vectors": vectors,
        }
        print(f"{pid}: loaded_items={len(items)}")

    embedder_lock = threading.Lock()
    platform_profiles = load_platform_profiles(profiles_path, platform_ids)
    # Persist initial empty profiles for reproducibility and explicit state tracking.
    persist_platform_profiles(profiles_path, platform_profiles)

    processed = 0
    skipped_no_intended = 0
    skipped_intended_not_hit = 0
    error_rounds = 0

    # Best-effort total for progress display (counts raw lines, not necessarily status=ok).
    with args.user_queries_path.open("r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    remaining_lines = max(0, total_lines - args.start_line + 1)
    total_target = remaining_lines if args.max_queries == 0 else min(remaining_lines, args.max_queries)

    with args.user_queries_path.open("r", encoding="utf-8") as src, rounds_path.open(
        "w", encoding="utf-8"
    ) as dst:
        line_no = 0
        with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
            while True:
                remaining_quota = args.max_concurrency
                if args.max_queries:
                    remaining_quota = min(remaining_quota, args.max_queries - processed)
                    if remaining_quota <= 0:
                        break

                batch_entries: List[Tuple[int, Dict]] = []
                while len(batch_entries) < remaining_quota:
                    line = src.readline()
                    if not line:
                        break
                    line_no += 1
                    if line_no < args.start_line:
                        continue

                    qrec = json.loads(line)
                    if qrec.get("status") != "ok":
                        continue
                    batch_entries.append((line_no, qrec))

                if not batch_entries:
                    break

                futures = [
                    executor.submit(
                        prepare_round,
                        qrec=qrec,
                        line_no=entry_line_no,
                        api=api,
                        chat_model=chat_model,
                        max_retries=args.max_retries,
                        sleep_seconds=args.sleep_seconds,
                        seed=args.seed,
                        intended_hit_prob=args.intended_hit_prob,
                        platform_ids=platform_ids,
                        platform_data=platform_data,
                        embedder=embedder,
                        embedder_lock=embedder_lock,
                    )
                    for entry_line_no, qrec in batch_entries
                ]

                for prepared in (future.result() for future in futures):
                    round_obj = prepared["round_obj"]

                    if prepared.get("candidates"):
                        try:
                            candidates: List[PlatformCandidate] = prepared["candidates"]
                            rank_list, rationale = ua_rank_candidates(
                                client=client,
                                model=chat_model,
                                max_retries=args.max_retries,
                                user_query=prepared["user_query"],
                                candidates=candidates,
                                platform_profiles=platform_profiles,
                                rng=make_stable_rng(args.seed, round_obj["query_id"], "ranking"),
                            )

                            candidate_map = {candidate.platform_id: candidate for candidate in candidates}
                            first_target_rank = next(
                                (
                                    idx
                                    for idx, pid in enumerate(rank_list, start=1)
                                    if candidate_map[pid].item.parent_asin == prepared["target_asin"]
                                ),
                                None,
                            )

                            if first_target_rank is None:
                                round_obj.update(
                                    {
                                        "status": "skipped_after_ranking_no_intended",
                                        "ua_rank_list": rank_list,
                                        "ua_rationale": rationale,
                                    }
                                )
                            else:
                                reward_pid = rank_list[first_target_rank - 1]
                                penalty_pids = rank_list[: first_target_rank - 1]

                                for pos, pid in enumerate(rank_list, start=1):
                                    event_type = None
                                    if pid in penalty_pids:
                                        event_type = "penalty"
                                    elif pid == reward_pid:
                                        event_type = "reward"
                                    if event_type is None:
                                        continue

                                    cand = candidate_map[pid]
                                    mem = generate_reputation_memory(
                                        client=client,
                                        model=chat_model,
                                        max_retries=args.max_retries,
                                        event_type=event_type,
                                        platform_id=pid,
                                        user_query=prepared["user_query"],
                                        item_title=cand.item.title,
                                        rank_pos=pos,
                                        target_rank=first_target_rank,
                                    )
                                    append_profile_memory(
                                        profiles=platform_profiles,
                                        platform_id=pid,
                                        entry=mem,
                                        window_size=args.profile_window_size,
                                    )

                                round_obj.update(
                                    {
                                        "status": "settled",
                                        "platform_candidates": [
                                            serialize_candidate(candidate) for candidate in candidates
                                        ],
                                        "intended_platform_ids": prepared["intended_platform_ids"],
                                        "ua_rank_list": rank_list,
                                        "ua_rationale": rationale,
                                        "target_rank": first_target_rank,
                                        "reward_platform": reward_pid,
                                        "penalty_platforms": penalty_pids,
                                    }
                                )
                        except Exception as err:
                            round_obj.update({"status": "error", "error": str(err)})

                    status = round_obj.get("status")
                    if status in {"skipped_no_intended", "skipped_after_ranking_no_intended"}:
                        skipped_no_intended += 1
                    elif status == "skipped_intended_not_hit":
                        skipped_intended_not_hit += 1
                    elif status == "error":
                        error_rounds += 1

                    dst.write(json.dumps(round_obj, ensure_ascii=False) + "\n")
                    processed += 1
                    persist_platform_profiles(profiles_path, platform_profiles)
                    sys.stderr.write(
                        f"\r[run_ua_pa_simulation] {processed}/{total_target} simulated | skipped_no_intended={skipped_no_intended} skipped_intended_not_hit={skipped_intended_not_hit} error_rounds={error_rounds}"
                    )
                    sys.stderr.flush()

    sys.stderr.write("\n")
    summary = {
        "user_queries_path": str(args.user_queries_path),
        "platform_dir": str(args.platform_dir),
        "rounds_path": str(rounds_path),
        "profiles_path": str(profiles_path),
        "processed_rounds": processed,
        "skipped_no_intended": skipped_no_intended,
        "skipped_intended_not_hit": skipped_intended_not_hit,
        "error_rounds": error_rounds,
        "chat_model": chat_model,
        "intended_hit_prob": args.intended_hit_prob,
        "profile_window_size": args.profile_window_size,
        "max_platform_items": args.max_platform_items,
        "embedding_model": args.embedding_model,
        "embedding_device": args.embedding_device,
        "embedding_batch_size": args.embedding_batch_size,
        "seed": args.seed,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    run_simulation(args)


if __name__ == "__main__":
    main()
