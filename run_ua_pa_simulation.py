#!/usr/bin/env python3
import argparse
import gc
import hashlib
import json
import math
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from openai import OpenAI
from config_loader import ApiConfig, load_api_config


PITCH_STYLE_GUIDES = {
    "honest": "Use a transparent, factual tone. Mention trade-offs when relevant, avoid hype, and do not stretch the fit.",
    "expert": "Use a calm, expert tone. Highlight the most decision-critical specs and who it's for, staying grounded in the item data.",
    "concise": "Be extremely concise: one sentence, no fluff, focus on the single strongest reason while staying grounded in the item data.",
    "auto": "Adapt the tone and selling strategy to the current item-query match. Decide whether to explain, reassure, or push harder.",
    "exaggerated": "Use an aggressively persuasive, overconfident tone that leans as hard as possible toward the user's query. You may stretch or selectively frame the fit if it helps make the item sound compelling.",
}


CACHE_FORMAT_VERSION = 1
RETRIEVAL_TEXT_VERSION = 1
NO_PURCHASE_DECISION = "NO_PURCHASE"


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


def item_to_cache_payload(item: Item) -> Dict:
    return {
        "parent_asin": item.parent_asin,
        "title": item.title,
        "main_category": item.main_category,
        "average_rating": item.average_rating,
        "rating_number": item.rating_number,
        "description": item.description,
        "reviews": item.reviews,
    }


def item_from_cache_payload(obj: Dict) -> Item:
    return Item(
        parent_asin=str(obj.get("parent_asin") or ""),
        title=clean_text(obj.get("title"), 300),
        main_category=clean_text(obj.get("main_category"), 120),
        average_rating=obj.get("average_rating"),
        rating_number=obj.get("rating_number"),
        description=clean_text(obj.get("description"), 1200),
        reviews=[clean_text(x, 220) for x in (obj.get("reviews") or []) if clean_text(x, 220)],
    )


def resolve_embedding_device(requested_device: str) -> str:
    requested = (requested_device or "auto").strip().lower()
    if requested != "auto":
        return requested

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"


def create_embedder(
    embedding_model: str,
    embedding_device: str,
    embedding_local_only: bool,
    embedding_max_seq_length: int,
    log_prefix: str = "",
):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as err:
        raise RuntimeError(
            "Missing dependency: sentence-transformers. Install with `pip install -U sentence-transformers`."
        ) from err

    try:
        embedder = SentenceTransformer(
            embedding_model,
            device=embedding_device,
            local_files_only=bool(embedding_local_only),
        )
    except TypeError:
        # Older sentence-transformers versions may not support local_files_only.
        if embedding_local_only:
            raise RuntimeError(
                "Your sentence-transformers version doesn't support local_files_only. "
                "Please upgrade with `pip install -U sentence-transformers`."
            )
        embedder = SentenceTransformer(embedding_model, device=embedding_device)

    if embedding_max_seq_length > 0:
        embedder.max_seq_length = int(embedding_max_seq_length)
        print(f"{log_prefix}Embedding max_seq_length set to {embedder.max_seq_length}")

    return embedder


def cleanup_embedder(embedder, embedding_device: str) -> None:
    del embedder
    gc.collect()
    try:
        import torch  # type: ignore

        if str(embedding_device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def parse_platform_embedding_devices(
    embedding_device: str,
    gpu_ids_text: str,
) -> List[str]:
    requested = str(gpu_ids_text or "").strip()
    if not requested:
        return [embedding_device]

    if not str(embedding_device).startswith("cuda"):
        print(
            f"Ignoring --platform-embedding-gpu-ids={requested!r} because embedding device resolved to {embedding_device}."
        )
        return [embedding_device]

    tokens = [tok.strip() for tok in requested.replace(" ", ",").split(",") if tok.strip()]
    if not tokens:
        return [embedding_device]

    devices: List[str] = []
    seen = set()
    for tok in tokens:
        try:
            gpu_id = int(tok)
        except ValueError as err:
            raise ValueError(f"Invalid GPU id in --platform-embedding-gpu-ids: {tok!r}") from err
        if gpu_id < 0:
            raise ValueError(f"GPU id must be >= 0, got {gpu_id}")
        device = f"cuda:{gpu_id}"
        if device not in seen:
            devices.append(device)
            seen.add(device)

    return devices or [embedding_device]


def assign_platforms_to_devices(
    platform_ids: List[str],
    platform_files: List[Path],
    devices: List[str],
) -> Dict[str, List[Tuple[str, Path]]]:
    assignments = {device: [] for device in devices}
    for idx, payload in enumerate(zip(platform_ids, platform_files)):
        device = devices[idx % len(devices)]
        assignments[device].append(payload)
    return assignments


def parse_platform_cache_dir(cache_dir_text: str) -> Optional[Path]:
    text = str(cache_dir_text or "").strip()
    if not text:
        return None
    return Path(text)


def platform_cache_paths(cache_dir: Path, platform_id: str) -> Dict[str, Path]:
    return {
        "meta": cache_dir / f"{platform_id}.meta.json",
        "items": cache_dir / f"{platform_id}.items.jsonl",
        "vectors": cache_dir / f"{platform_id}.vectors.npy",
    }


def build_platform_cache_metadata(
    platform_id: str,
    platform_file: Path,
    max_items: int,
    embedding_model: str,
    embedding_max_seq_length: int,
    item_count: int,
    vector_dim: int,
) -> Dict:
    stat = platform_file.stat()
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "retrieval_text_version": RETRIEVAL_TEXT_VERSION,
        "platform_id": platform_id,
        "platform_file": str(platform_file.resolve()),
        "platform_file_size": stat.st_size,
        "platform_file_mtime_ns": stat.st_mtime_ns,
        "max_items": int(max_items),
        "embedding_model": embedding_model,
        "embedding_max_seq_length": int(embedding_max_seq_length),
        "item_count": int(item_count),
        "vector_dim": int(vector_dim),
    }


def write_json_atomic(path: Path, obj: Dict) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def write_jsonl_atomic(path: Path, rows: Iterable[Dict]) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def save_platform_cache(
    cache_dir: Path,
    platform_id: str,
    platform_file: Path,
    items: List[Item],
    vectors,
    max_items: int,
    embedding_model: str,
    embedding_max_seq_length: int,
) -> None:
    import numpy as np

    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = platform_cache_paths(cache_dir, platform_id)
    meta = build_platform_cache_metadata(
        platform_id=platform_id,
        platform_file=platform_file,
        max_items=max_items,
        embedding_model=embedding_model,
        embedding_max_seq_length=embedding_max_seq_length,
        item_count=len(items),
        vector_dim=int(vectors.shape[1]) if getattr(vectors, "ndim", 0) == 2 else 0,
    )

    vectors_tmp = paths["vectors"].with_name(paths["vectors"].name + ".tmp")
    with vectors_tmp.open("wb") as f:
        np.save(f, np.asarray(vectors, dtype=np.float32), allow_pickle=False)

    write_jsonl_atomic(paths["items"], (item_to_cache_payload(item) for item in items))
    vectors_tmp.replace(paths["vectors"])
    write_json_atomic(paths["meta"], meta)


def load_platform_cache(
    cache_dir: Path,
    platform_id: str,
    platform_file: Path,
    max_items: int,
    embedding_model: str,
    embedding_max_seq_length: int,
) -> Optional[Tuple[List[Item], Dict[str, Item], "np.ndarray"]]:
    import numpy as np

    paths = platform_cache_paths(cache_dir, platform_id)
    if not all(path.exists() for path in paths.values()):
        print(f"{platform_id}: cache miss, rebuilding")
        return None

    try:
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    except Exception as err:
        print(f"{platform_id}: cache meta unreadable ({err}), rebuilding")
        return None

    expected_meta = build_platform_cache_metadata(
        platform_id=platform_id,
        platform_file=platform_file,
        max_items=max_items,
        embedding_model=embedding_model,
        embedding_max_seq_length=embedding_max_seq_length,
        item_count=int(meta.get("item_count") or 0),
        vector_dim=int(meta.get("vector_dim") or 0),
    )
    for key in (
        "cache_format_version",
        "retrieval_text_version",
        "platform_id",
        "platform_file",
        "platform_file_size",
        "platform_file_mtime_ns",
        "max_items",
        "embedding_model",
        "embedding_max_seq_length",
    ):
        if meta.get(key) != expected_meta.get(key):
            print(f"{platform_id}: cache invalid ({key} mismatch), rebuilding")
            return None

    try:
        items: List[Item] = []
        asin_map: Dict[str, Item] = {}
        with paths["items"].open("r", encoding="utf-8") as f:
            for line in f:
                item = item_from_cache_payload(json.loads(line))
                items.append(item)
                asin_map[item.parent_asin] = item

        with paths["vectors"].open("rb") as f:
            vectors = np.load(f, allow_pickle=False)
        vectors = np.asarray(vectors, dtype=np.float32)
    except Exception as err:
        print(f"{platform_id}: cache payload unreadable ({err}), rebuilding")
        return None

    if len(items) != int(meta.get("item_count") or -1):
        print(f"{platform_id}: cache invalid (item_count mismatch), rebuilding")
        return None
    if getattr(vectors, "ndim", 0) != 2 or vectors.shape[0] != len(items):
        print(f"{platform_id}: cache invalid (vector shape mismatch), rebuilding")
        return None
    if len(items) and vectors.shape[1] != int(meta.get("vector_dim") or -1):
        print(f"{platform_id}: cache invalid (vector_dim mismatch), rebuilding")
        return None

    print(f"{platform_id}: loaded_items={len(items)} from cache")
    return items, asin_map, vectors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run UA-PA simulation: structured query, platform recall, UA ranking, settlement."
    )
    parser.add_argument(
        "--user-queries-path",
        type=Path,
        default=Path("./output/user_queries.jsonl"),
        help="Path to generated user query JSONL",
    )
    parser.add_argument(
        "--platform-dir",
        type=Path,
        default=Path("./output/platforms"),
        help="Directory containing platform_1.jsonl ... platform_5.jsonl",
    )
    parser.add_argument(
        "--platform-cache-dir",
        type=str,
        default="./output/platform_cache",
        help="Optional cache directory for platform retrieval vectors. Empty means rebuild embeddings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output/sim"),
        help="Output directory for simulation artifacts",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=50,
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
        default=0.1,
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
        default=30000,
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
        default="auto",
        help="Device for embedding model: auto, cpu, cuda, or mps",
    )
    parser.add_argument(
        "--platform-embedding-gpu-ids",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated GPU ids for platform catalog embedding, e.g. 0,1,2. If empty, use a single resolved embedding device.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=1024,
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
        default=True,
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
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=10,
        help="Retrieve top-K items per platform before sampling one with Gumbel-Max",
    )
    parser.add_argument(
        "--gumbel-temperature",
        type=float,
        default=0.1,
        help="Temperature for Gumbel-Max retrieval sampling; lower means closer to greedy top-1",
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


def format_history_for_ua(history_items: Sequence[Dict]) -> str:
    if not history_items:
        return "None"

    lines: List[str] = []
    for idx, item in enumerate(history_items, start=1):
        parts = [f"title={clean_text(item.get('title'), 120) or 'Unknown title'}"]
        rating = item.get("rating")
        timestamp = item.get("timestamp")
        review = clean_text(item.get("text"), 180)
        if rating is not None:
            parts.append(f"rating={rating}")
        if timestamp:
            parts.append(f"timestamp={timestamp}")
        if review:
            parts.append(f"review={review}")
        lines.append(f"{idx}. " + " | ".join(parts))
    return "\n".join(lines)


def fallback_long_term_need(history_items: Sequence[Dict], query: str) -> str:
    titles = [
        clean_text(item.get("title"), 80)
        for item in history_items[-3:]
        if clean_text(item.get("title"), 80)
    ]
    if titles:
        return clean_text("Long-term interest around " + "; ".join(titles), 180)
    return clean_text(query, 180)


def ua_long_term_need(ua_structured: Dict) -> str:
    return clean_text(str(ua_structured.get("long_term_need") or ""), 180)


def ua_current_need(ua_structured: Dict) -> str:
    return clean_text(str(ua_structured.get("current_need") or ""), 180)


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
        raw = obj["ranked_platform_ids"] if "ranked_platform_ids" in obj else obj.get("rank_list")
        decision = str(obj.get("decision") or "").strip().upper()
        no_purchase_flag = str(obj.get("no_purchase") or "").strip().lower() == "true"
        if (no_purchase_flag or decision == NO_PURCHASE_DECISION) and raw == []:
            return []
        if isinstance(raw, list):
            rank = [str(x) for x in raw]
            if sorted(rank) == sorted(valid_ids):
                return rank

    if re.search(rf"\b{NO_PURCHASE_DECISION}\b", text):
        return []

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
        "P1": "concise",
        "P2": "expert",
        "P3": "honest",
        "P4": "exaggerated",
        "P5": "auto",
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


def load_platform_items_group_worker(
    embedding_device: str,
    assigned_platforms: List[Tuple[str, Path]],
    embedder,
    max_items: int,
    embedding_batch_size: int,
    embedding_show_progress: bool,
) -> Dict[str, Tuple[List[Item], Dict[str, Item], "np.ndarray"]]:
    results: Dict[str, Tuple[List[Item], Dict[str, Item], "np.ndarray"]] = {}
    for pid, platform_file in assigned_platforms:
        print(f"{pid}: assigned_device={embedding_device}")
        print(f"{pid}: reading_items+embedding from {platform_file} ...")
        items, asin_map, vectors = load_platform_items(
            platform_file=platform_file,
            max_items=max_items,
            embedder=embedder,
            embedding_batch_size=embedding_batch_size,
            embedding_show_progress=embedding_show_progress,
        )
        print(f"{pid}: loaded_items={len(items)}")
        results[pid] = (items, asin_map, vectors)

    return results


def ua_structure_query(
    client: OpenAI,
    model: str,
    max_retries: int,
    query_record: Dict,
) -> Dict:
    query = query_record.get("query_text", "")
    style = query_record.get("query_style", "")
    history_items = query_record.get("history_items") or []

    user_prompt = f"""
Convert the following user query into a compact JSON object for downstream retrieval.
Use the user's past history to infer longer-term preference, while keeping the
current intent tightly grounded in the current query.
Output JSON only (no markdown, no extra keys).

Schema (types + limits):
- long_term_need: string (<= 30 words). Reflect the user's broader, more stable need or preference pattern, inferred mainly from history.
- current_need: string (<= 25 words). Rewrite the current query for retrieval. Keep meaning; do NOT add new brands/models/specs not implied by the query.

Rules:
- long_term_need should mainly come from history rather than simply paraphrasing the current query, unless history is empty or uninformative.
- current_need should stay close to the current query and capture the user's immediate task.

Example output:
{{
  "long_term_need": "Beginner-friendly home music gear with low setup burden",
  "current_need": "beginner electric guitar starter kit for home practice"
}}

User history:
{format_history_for_ua(history_items)}

Current query:
query_text={query}
""".strip()

    system = "You are a strict JSON generator for recommendation query understanding."

    fallback = {
        "long_term_need": fallback_long_term_need(history_items, query),
        "current_need": clean_text(str(query), 180),
        "style": style,
    }

    try:
        raw = llm_chat_json(client, model, system, user_prompt, max_retries)
        obj = parse_json_object(raw)
        if obj:
            return {
                "long_term_need": clean_text(
                    str(obj.get("long_term_need") or fallback["long_term_need"]),
                    180,
                ),
                "current_need": clean_text(
                    str(obj.get("current_need") or fallback["current_need"]),
                    180,
                ),
                "style": style,
            }
    except Exception:
        pass

    return fallback


def build_query_embedding_text(ua_structured: Dict) -> str:
    current_need = ua_current_need(ua_structured)
    long_term_need = ua_long_term_need(ua_structured)
    parts = [current_need]
    if long_term_need and long_term_need != current_need:
        parts.append(long_term_need)
    return " ".join(parts)


def encode_query_vector(ua_structured: Dict, embedder):
    import numpy as np

    qvec = embedder.encode(
        [build_query_embedding_text(ua_structured)],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(qvec, dtype=np.float32).reshape(-1)


def sample_gumbel_noise(rng: random.Random) -> float:
    u = min(max(rng.random(), 1e-12), 1.0 - 1e-12)
    return -math.log(-math.log(u))


def retrieve_topk_gumbel(
    items: List[Item],
    vectors,
    qvec,
    top_k: int,
    gumbel_temperature: float,
    rng: random.Random,
) -> Tuple[Item, float]:
    import numpy as np

    if not items:
        raise ValueError("Platform has no items loaded")
    if vectors is None or getattr(vectors, "shape", (0,))[0] == 0:
        raise ValueError("Platform has no embedding vectors loaded")

    scores = vectors @ qvec
    candidate_count = min(max(1, int(top_k)), len(items))
    if candidate_count == 1:
        best_idx = int(scores.argmax())
        return items[best_idx], float(scores[best_idx])

    score_count = int(scores.shape[0])
    if candidate_count >= score_count:
        top_idx = np.argsort(scores)[::-1]
    else:
        partition_idx = score_count - candidate_count
        top_idx_unsorted = np.argpartition(scores, partition_idx)[partition_idx:]
        top_idx = top_idx_unsorted[np.argsort(scores[top_idx_unsorted])[::-1]]
    scaled_scores = scores[top_idx] / float(gumbel_temperature)
    noisy_scores = np.asarray(
        [float(score) + sample_gumbel_noise(rng) for score in scaled_scores],
        dtype=np.float32,
    )
    chosen_pos = int(noisy_scores.argmax())
    chosen_idx = int(top_idx[chosen_pos])
    return items[chosen_idx], float(scores[chosen_idx])


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
for the candidate item based on the user's long-term preference and current intent.

Pitch style: {pitch_style}
Style guidance: {style_instruction}

Requirements:
- 1 to 3 sentences
- Tie the message to user needs and usage scenario
- Return plain text only

User preference and current intent:
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


def build_platform_candidate(
    api: ApiConfig,
    chat_model: str,
    max_retries: int,
    pid: str,
    ua_struct: Dict,
    target_asin: str,
    intended_hit_pids: List[str],
    platform_data: Dict[str, Dict],
    qvec,
    retrieval_top_k: int,
    gumbel_temperature: float,
    retrieval_rng: random.Random,
) -> PlatformCandidate:
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
        chosen, score = retrieve_topk_gumbel(
            items=items,
            vectors=vectors,
            qvec=qvec,
            top_k=retrieval_top_k,
            gumbel_temperature=gumbel_temperature,
            rng=retrieval_rng,
        )

    style = choose_pitch_style(pid)
    pitch = generate_platform_pitch(
        client=make_client(api),
        model=chat_model,
        max_retries=max_retries,
        platform_id=pid,
        ua_structured=ua_struct,
        item=chosen,
        pitch_style=style,
    )
    return PlatformCandidate(
        platform_id=pid,
        item=chosen,
        pitch=pitch,
        pitch_style=style,
        retrieval_score=score,
        forced_intended_hit=forced_hit,
    )


def parallel_generate_platform_candidates(
    api: ApiConfig,
    chat_model: str,
    max_retries: int,
    platform_ids: List[str],
    ua_struct: Dict,
    target_asin: str,
    intended_hit_pids: List[str],
    platform_data: Dict[str, Dict],
    qvec,
    seed: int,
    query_id: str,
    retrieval_top_k: int,
    gumbel_temperature: float,
) -> List[PlatformCandidate]:
    with ThreadPoolExecutor(max_workers=max(1, len(platform_ids))) as executor:
        futures = [
            executor.submit(
                build_platform_candidate,
                api,
                chat_model,
                max_retries,
                pid,
                ua_struct,
                target_asin,
                intended_hit_pids,
                platform_data,
                qvec,
                retrieval_top_k,
                gumbel_temperature,
                make_stable_rng(seed, query_id, pid, "retrieval_gumbel"),
            )
            for pid in platform_ids
        ]
        return [future.result() for future in futures]


def build_target_item_payload(target_item: Dict) -> Dict:
    return {
        "parent_asin": str(target_item.get("parent_asin") or ""),
        "title": clean_text(target_item.get("title"), 300),
        "rating": target_item.get("rating"),
        "user_review": clean_text(target_item.get("text"), 280),
    }


def generate_purchase_reputation_memory(
    client: OpenAI,
    model: str,
    max_retries: int,
    user_query: str,
    target_item: Dict,
    purchased_candidate: PlatformCandidate,
) -> str:
    target_payload = build_target_item_payload(target_item)
    purchased_item_payload = purchased_candidate.item.to_pitch_payload()
    prompt = f"""
You are updating a platform reputation memory after a simulated purchase.
Write ONE or TWO short English sentences (total <= 55 words) that will help future ranking.

The user saw the platform pitch before buying, then experienced the item's real information after purchase.
Compare both against what the user likely wanted, represented by the real next item.

User query:
{user_query}

Real next item the user actually wanted:
{json.dumps(target_payload, ensure_ascii=False)}

Purchased item real information:
{json.dumps(purchased_item_payload, ensure_ascii=False)}

Platform pitch shown before purchase:
{purchased_candidate.pitch}

Rules:
- Judge whether the platform pitch or the purchased item's real information is closer to the user's intended item.
- The memory may be positive, negative, or mixed depending on the evidence.
- If alignment is strong, write a positive trust signal only.
- Only mention a mismatch risk when there is a concrete unsupported claim, incompatibility, or intent mismatch.
- If evidence is mixed, you may mention both one trust signal and one risk.
- Prefer specific factual observations over generic cautionary language.
- Do not invent specs, reviews, or guarantees.
- Output plain text only.
""".strip()
    try:
        return clean_text(
            llm_chat_json(
                client=client,
                model=model,
                system_prompt="You create concise post-purchase platform reputation memories.",
                user_prompt=prompt,
                max_retries=max_retries,
            ),
            320,
        )
    except Exception:
        target_title = clean_text(target_item.get("title"), 120) or "the intended item"
        purchased_title = clean_text(purchased_candidate.item.title, 120) or "the purchased item"
        if target_title.lower() != purchased_title.lower():
            return (
                f"The top-ranked pitch led the user toward {purchased_title} instead of {target_title}. "
                "Post-purchase details suggest possible intent mismatch risk."
            )
        return (
            "The top-ranked pitch stayed close to the user's likely need. "
            "Post-purchase details look broadly consistent, so this platform seems reasonably trustworthy."
        )


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
    retrieval_top_k: int,
    gumbel_temperature: float,
) -> Dict:
    user_query = str(qrec.get("query_text") or "")
    target_item = qrec.get("target_item") or {}
    target_asin = str(target_item.get("parent_asin") or "")
    query_id = str(qrec.get("query_id") or f"line_{line_no}")
    round_obj = {
        "query_id": query_id,
        "source_line": line_no,
        "user_id": qrec.get("user_id"),
        "query_text": user_query,
        "target_asin": target_asin,
        "target_item": build_target_item_payload(target_item),
        "status": "started",
    }

    try:
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        ua_struct = ua_structure_query(
            client=make_client(api),
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
        round_obj["intended_item_exists_in_any_platform"] = intended_exists_any
        round_obj["intended_hit_platform_ids"] = intended_hit_pids

        qvec = encode_query_vector(ua_struct, embedder)
        candidates = parallel_generate_platform_candidates(
            api=api,
            chat_model=chat_model,
            max_retries=max_retries,
            platform_ids=platform_ids,
            ua_struct=ua_struct,
            target_asin=target_asin,
            intended_hit_pids=intended_hit_pids,
            platform_data=platform_data,
            qvec=qvec,
            seed=seed,
            query_id=query_id,
            retrieval_top_k=retrieval_top_k,
            gumbel_temperature=gumbel_temperature,
        )

        intended_platforms = [
            candidate.platform_id
            for candidate in candidates
            if candidate.item.parent_asin == target_asin
        ]

        return {
            "round_obj": round_obj,
            "user_query": user_query,
            "target_asin": target_asin,
            "target_item": target_item,
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
If none of the candidates is a good purchase, output this exact decision format instead:
{{"decision":"{NO_PURCHASE_DECISION}","no_purchase":true,"ranked_platform_ids":[],"rationale":"brief explanation"}}
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
    started_at = time.perf_counter()

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
    if args.retrieval_top_k <= 0:
        raise ValueError("--retrieval-top-k must be > 0")
    if args.gumbel_temperature <= 0:
        raise ValueError("--gumbel-temperature must be > 0")

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

    requested_embedding_device = args.embedding_device
    embedding_device = resolve_embedding_device(requested_embedding_device)
    args.embedding_device = embedding_device
    if str(requested_embedding_device).strip().lower() == "auto":
        print(f"Resolved embedding device automatically: {embedding_device}")
    platform_cache_dir = parse_platform_cache_dir(args.platform_cache_dir)
    if platform_cache_dir is not None:
        print(f"Platform cache directory: {platform_cache_dir}")

    platform_data = {}
    cache_hits = 0
    cache_misses = 0
    pending_platform_entries: List[Tuple[str, Path]] = []
    for pid, pfile in zip(platform_ids, platform_files):
        cached = None
        if platform_cache_dir is not None:
            cached = load_platform_cache(
                cache_dir=platform_cache_dir,
                platform_id=pid,
                platform_file=pfile,
                max_items=args.max_platform_items,
                embedding_model=args.embedding_model,
                embedding_max_seq_length=args.embedding_max_seq_length,
            )
        if cached is not None:
            items, asin_map, vectors = cached
            platform_data[pid] = {
                "items": items,
                "asin_map": asin_map,
                "vectors": vectors,
            }
            cache_hits += 1
        else:
            pending_platform_entries.append((pid, pfile))
            cache_misses += 1
    pending_platform_map = dict(pending_platform_entries)

    platform_embedding_devices = parse_platform_embedding_devices(
        embedding_device=embedding_device,
        gpu_ids_text=args.platform_embedding_gpu_ids,
    )
    platform_assignments = assign_platforms_to_devices(
        platform_ids=[pid for pid, _ in pending_platform_entries],
        platform_files=[pfile for _, pfile in pending_platform_entries],
        devices=platform_embedding_devices,
    )
    if pending_platform_entries:
        print(
            "Loading platform catalogs and building retrieval vectors "
            f"for {len(pending_platform_entries)} cache-miss platform(s) across {len(platform_embedding_devices)} device(s)..."
        )
        for device in platform_embedding_devices:
            assigned_pids = [pid for pid, _ in platform_assignments[device]]
            print(f"{device}: assigned_platforms={assigned_pids}")

        platform_embedders = {}
        for device in platform_embedding_devices:
            assigned_pids = [pid for pid, _ in platform_assignments[device]]
            if not assigned_pids:
                continue
            print(f"{device}: loading embedding model replica for {assigned_pids} ...")
            platform_embedders[device] = create_embedder(
                embedding_model=args.embedding_model,
                embedding_device=device,
                embedding_local_only=bool(args.embedding_local_only),
                embedding_max_seq_length=args.embedding_max_seq_length,
                log_prefix=f"{device}: ",
            )

        try:
            with ThreadPoolExecutor(max_workers=len(platform_embedders)) as executor:
                future_to_device = {
                    executor.submit(
                        load_platform_items_group_worker,
                        embedding_device=device,
                        assigned_platforms=platform_assignments[device],
                        embedder=platform_embedders[device],
                        max_items=args.max_platform_items,
                        embedding_batch_size=args.embedding_batch_size,
                        embedding_show_progress=args.embedding_show_progress,
                    ): device
                    for device in platform_embedders
                }
                for future in as_completed(future_to_device):
                    device = future_to_device[future]
                    group_results = future.result()
                    embedder_to_release = platform_embedders.pop(device, None)
                    if embedder_to_release is not None:
                        cleanup_embedder(embedder_to_release, device)
                    for pid, result in group_results.items():
                        items, asin_map, vectors = result
                        if not items:
                            raise RuntimeError(f"{pid} has no loaded items on {device}")
                        platform_data[pid] = {
                            "items": items,
                            "asin_map": asin_map,
                            "vectors": vectors,
                        }
                        if platform_cache_dir is not None:
                            save_platform_cache(
                                cache_dir=platform_cache_dir,
                                platform_id=pid,
                                platform_file=pending_platform_map[pid],
                                items=items,
                                vectors=vectors,
                                max_items=args.max_platform_items,
                                embedding_model=args.embedding_model,
                                embedding_max_seq_length=args.embedding_max_seq_length,
                            )
                            print(f"{pid}: cache saved to {platform_cache_dir}")
        finally:
            for device, embedder_to_release in list(platform_embedders.items()):
                cleanup_embedder(embedder_to_release, device)
                platform_embedders.pop(device, None)
    else:
        print("All platform catalogs loaded from cache.")

    if len(platform_data) != len(platform_ids):
        missing = [pid for pid in platform_ids if pid not in platform_data]
        raise RuntimeError(f"Failed to load platform data for: {missing}")

    query_embedding_device = platform_embedding_devices[0] if platform_embedding_devices else embedding_device
    print(f"Loading query embedding model: {args.embedding_model} (device={query_embedding_device})")
    embedder = create_embedder(
        embedding_model=args.embedding_model,
        embedding_device=query_embedding_device,
        embedding_local_only=bool(args.embedding_local_only),
        embedding_max_seq_length=args.embedding_max_seq_length,
    )

    platform_profiles = load_platform_profiles(profiles_path, platform_ids)
    # Persist initial empty profiles for reproducibility and explicit state tracking.
    persist_platform_profiles(profiles_path, platform_profiles)

    processed = 0
    skipped_no_intended = 0
    skipped_intended_not_hit = 0
    error_rounds = 0
    settled_rounds = 0
    target_missing_after_ranking = 0
    profile_updates_written = 0

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
                        retrieval_top_k=args.retrieval_top_k,
                        gumbel_temperature=args.gumbel_temperature,
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
                            purchased_pid = rank_list[0] if rank_list else ""
                            purchased_candidate = candidate_map.get(purchased_pid)
                            reputation_memory = ""
                            purchased_item = {}
                            purchased_pitch = ""
                            purchased_pitch_style = ""
                            if purchased_candidate is not None:
                                reputation_memory = generate_purchase_reputation_memory(
                                    client=client,
                                    model=chat_model,
                                    max_retries=args.max_retries,
                                    user_query=prepared["user_query"],
                                    target_item=prepared["target_item"],
                                    purchased_candidate=purchased_candidate,
                                )
                                append_profile_memory(
                                    profiles=platform_profiles,
                                    platform_id=purchased_pid,
                                    entry=reputation_memory,
                                    window_size=args.profile_window_size,
                                )
                                purchased_item = purchased_candidate.item.to_pitch_payload()
                                purchased_pitch = purchased_candidate.pitch
                                purchased_pitch_style = purchased_candidate.pitch_style

                            round_obj.update(
                                {
                                    "status": "settled",
                                    "platform_candidates": [
                                        serialize_candidate(candidate) for candidate in candidates
                                    ],
                                    "intended_platform_ids": prepared["intended_platform_ids"],
                                    "ua_rank_list": rank_list,
                                    "ua_rationale": rationale,
                                    "no_purchase": purchased_candidate is None,
                                    "target_rank": first_target_rank,
                                    "reward_platform": purchased_pid,
                                    "purchased_platform": purchased_pid,
                                    "purchased_item": purchased_item,
                                    "purchased_pitch": purchased_pitch,
                                    "purchased_pitch_style": purchased_pitch_style,
                                    "penalty_platforms": [],
                                    "profile_memory_update": reputation_memory,
                                }
                            )
                        except Exception as err:
                            round_obj.update({"status": "error", "error": str(err)})

                    status = round_obj.get("status")
                    if status == "skipped_no_intended":
                        skipped_no_intended += 1
                    elif status == "skipped_intended_not_hit":
                        skipped_intended_not_hit += 1
                    elif status == "settled":
                        settled_rounds += 1
                        target_rank_raw = round_obj.get("target_rank")
                        target_rank = (
                            int(target_rank_raw)
                            if isinstance(target_rank_raw, int)
                            or str(target_rank_raw or "").isdigit()
                            else 0
                        )
                        if target_rank <= 0:
                            target_missing_after_ranking += 1
                        if str(round_obj.get("profile_memory_update") or "").strip():
                            profile_updates_written += 1
                    elif status == "error":
                        error_rounds += 1

                    dst.write(json.dumps(round_obj, ensure_ascii=False) + "\n")
                    processed += 1
                    persist_platform_profiles(profiles_path, platform_profiles)
                    sys.stderr.write(
                        f"\r[run_ua_pa_simulation] {processed}/{total_target} simulated | settled={settled_rounds} target_missing_after_ranking={target_missing_after_ranking} profile_updates={profile_updates_written} error_rounds={error_rounds}"
                    )
                    sys.stderr.flush()

    sys.stderr.write("\n")
    elapsed_seconds = time.perf_counter() - started_at
    summary = {
        "user_queries_path": str(args.user_queries_path),
        "platform_dir": str(args.platform_dir),
        "rounds_path": str(rounds_path),
        "profiles_path": str(profiles_path),
        "processed_rounds": processed,
        "settled_rounds": settled_rounds,
        "skipped_no_intended": skipped_no_intended,
        "skipped_intended_not_hit": skipped_intended_not_hit,
        "target_missing_after_ranking": target_missing_after_ranking,
        "profile_updates_written": profile_updates_written,
        "error_rounds": error_rounds,
        "chat_model": chat_model,
        "intended_hit_prob": args.intended_hit_prob,
        "profile_window_size": args.profile_window_size,
        "max_platform_items": args.max_platform_items,
        "embedding_model": args.embedding_model,
        "embedding_device": args.embedding_device,
        "platform_cache_dir": str(platform_cache_dir) if platform_cache_dir is not None else "",
        "platform_cache_hits": cache_hits,
        "platform_cache_misses": cache_misses,
        "platform_embedding_devices": platform_embedding_devices,
        "query_embedding_device": query_embedding_device,
        "embedding_batch_size": args.embedding_batch_size,
        "retrieval_top_k": args.retrieval_top_k,
        "gumbel_temperature": args.gumbel_temperature,
        "seed": args.seed,
        "elapsed_seconds": elapsed_seconds,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[run_ua_pa_simulation] elapsed_seconds={elapsed_seconds:.2f}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    run_simulation(args)


if __name__ == "__main__":
    main()
