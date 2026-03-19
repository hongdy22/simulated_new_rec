#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from heapq import nlargest
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI
from config_loader import ApiConfig, load_api_config


PITCH_STYLE_GUIDES = {
    "honest": "Use a transparent, factual tone. Mention trade-offs when relevant and avoid hype.",
    "exaggerated": "Use a promotional, high-energy tone while staying plausible and not inventing specs.",
    "unconstrained": "No extra style constraints. Write naturally in your own preferred persuasive style.",
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


@dataclass
class BM25Index:
    postings: Dict[str, List[Tuple[int, int]]]  # token -> [(doc_id, tf)]
    doc_len: List[int]
    idf: Dict[str, float]
    avgdl: float
    k1: float
    b: float

    def topk(self, query: str, k: int) -> List[Tuple[int, float]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores: Dict[int, float] = {}
        for tok in q_tokens:
            plist = self.postings.get(tok)
            if not plist:
                continue
            idf = self.idf.get(tok, 0.0)
            for doc_id, tf in plist:
                dl = self.doc_len[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                s = idf * (tf * (self.k1 + 1.0)) / (denom if denom else 1.0)
                scores[doc_id] = scores.get(doc_id, 0.0) + s

        if not scores:
            return []
        return nlargest(k, scores.items(), key=lambda x: x[1])


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
        "--hash-embedding-dim",
        type=int,
        default=512,
        help="Dimension for hash-based embedding retrieval",
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.45,
        help="Hybrid score weight for BM25 vs vector (alpha*bm25 + (1-alpha)*vector)",
    )
    parser.add_argument(
        "--hybrid-topk",
        type=int,
        default=200,
        help="BM25 candidate pool size for hybrid retrieval (topK by BM25)",
    )
    parser.add_argument(
        "--bm25-k1",
        type=float,
        default=1.2,
        help="BM25 k1 parameter",
    )
    parser.add_argument(
        "--bm25-b",
        type=float,
        default=0.75,
        help="BM25 b parameter",
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
    return parser.parse_args()


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


def choose_pitch_style(seed_text: str) -> str:
    styles = sorted(PITCH_STYLE_GUIDES.keys())
    digest = hashlib.md5(seed_text.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(styles)
    return styles[idx]


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


def build_bm25_index(
    docs: List[str],
    k1: float,
    b: float,
) -> BM25Index:
    postings: Dict[str, List[Tuple[int, int]]] = {}
    df: Dict[str, int] = {}
    doc_len: List[int] = []

    for doc_id, text in enumerate(docs):
        toks = tokenize(text)
        doc_len.append(len(toks))
        if not toks:
            continue
        tf_map: Dict[str, int] = {}
        for t in toks:
            tf_map[t] = tf_map.get(t, 0) + 1
        for t, tf in tf_map.items():
            postings.setdefault(t, []).append((doc_id, tf))
        for t in tf_map.keys():
            df[t] = df.get(t, 0) + 1

    n_docs = max(1, len(docs))
    avgdl = sum(doc_len) / n_docs if doc_len else 1.0

    idf: Dict[str, float] = {}
    for t, dft in df.items():
        # BM25+ style IDF, stable for rare/very common terms
        idf[t] = math.log(1.0 + (n_docs - dft + 0.5) / (dft + 0.5))

    return BM25Index(
        postings=postings,
        doc_len=doc_len if doc_len else [1] * n_docs,
        idf=idf,
        avgdl=avgdl if avgdl > 0 else 1.0,
        k1=k1,
        b=b,
    )


def load_platform_items(
    platform_file: Path,
    max_items: int,
    emb_dim: int,
    bm25_k1: float,
    bm25_b: float,
) -> Tuple[List[Item], Dict[str, Item], List[List[float]], BM25Index]:
    items: List[Item] = []
    asin_map: Dict[str, Item] = {}
    vectors: List[List[float]] = []
    doc_texts: List[str] = []

    with platform_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if max_items and len(items) >= max_items:
                break
            obj = json.loads(line)
            item = build_item_from_obj(obj)
            if not item.parent_asin:
                continue
            text = item_retrieval_text(item)
            vec = hash_embedding(text, emb_dim)
            items.append(item)
            asin_map[item.parent_asin] = item
            vectors.append(vec)
            doc_texts.append(text)

    bm25 = build_bm25_index(doc_texts, k1=bm25_k1, b=bm25_b)
    return items, asin_map, vectors, bm25


def ua_structure_query(
    client: OpenAI,
    model: str,
    max_retries: int,
    query_record: Dict,
) -> Dict:
    query = query_record.get("query_text", "")
    style = query_record.get("query_style", "")

    user_prompt = f"""
Convert the following user query into a structured JSON object for downstream
platform retrieval.
Output JSON only. No markdown, no extra commentary.

Required fields:
- user_need: core user need as a short sentence
- query_rewrite: retrieval-oriented rewritten query
- keywords: array of 3-8 keywords
- constraints: array of 0-5 constraints (compatibility, budget, usage context, etc.)
- style: keep the original user expression style

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


def retrieve_top1(
    ua_structured: Dict,
    items: List[Item],
    vectors: List[List[float]],
    emb_dim: int,
    bm25: BM25Index,
    hybrid_alpha: float,
    hybrid_topk: int,
) -> Tuple[Item, float]:
    if not items:
        raise ValueError("Platform has no items loaded")

    qtext = " ".join(
        [
            str(ua_structured.get("query_rewrite") or ""),
            " ".join(str(x) for x in (ua_structured.get("keywords") or [])),
            " ".join(str(x) for x in (ua_structured.get("constraints") or [])),
            str(ua_structured.get("user_need") or ""),
        ]
    )
    qvec = hash_embedding(qtext, emb_dim)

    # Stage 1: BM25 candidate pool (no full-scan)
    candidates = bm25.topk(qtext, k=max(1, hybrid_topk))
    if not candidates:
        # Fallback to full scan on hash vectors only if BM25 can't score anything.
        best_idx = 0
        best_score = -1e9
        for idx, ivec in enumerate(vectors):
            s = cosine(qvec, ivec)
            if s > best_score:
                best_score = s
                best_idx = idx
        return items[best_idx], best_score

    bm25_scores = [s for _, s in candidates]
    bmin, bmax = min(bm25_scores), max(bm25_scores)
    if bmax > bmin:
        bm25_norm = {doc_id: (s - bmin) / (bmax - bmin) for doc_id, s in candidates}
    else:
        bm25_norm = {doc_id: 0.0 for doc_id, _ in candidates}

    vec_raw: Dict[int, float] = {}
    for doc_id, _ in candidates:
        vec_raw[doc_id] = cosine(qvec, vectors[doc_id])
    vmin, vmax = min(vec_raw.values()), max(vec_raw.values())
    if vmax > vmin:
        vec_norm = {doc_id: (s - vmin) / (vmax - vmin) for doc_id, s in vec_raw.items()}
    else:
        vec_norm = {doc_id: 0.0 for doc_id, _ in candidates}

    alpha = float(hybrid_alpha)
    if alpha < 0.0:
        alpha = 0.0
    if alpha > 1.0:
        alpha = 1.0

    best_idx = candidates[0][0]
    best_score = -1e9
    for doc_id, _ in candidates:
        s = alpha * bm25_norm.get(doc_id, 0.0) + (1.0 - alpha) * vec_norm.get(doc_id, 0.0)
        if s > best_score:
            best_score = s
            best_idx = doc_id
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


def ua_rank_candidates(
    client: OpenAI,
    model: str,
    max_retries: int,
    user_query: str,
    candidates: List[PlatformCandidate],
    platform_profiles: Dict[str, List[str]],
) -> Tuple[List[str], str]:
    shuffled = candidates[:]
    random.shuffle(shuffled)

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
You are a platform reputation logger. Generate one short memory note in English
(max 20 words). Output one sentence only.

event_type={event_type}
platform_id={platform_id}
user_query={user_query}
item_title={item_title}
rank_pos={rank_pos}
target_rank={target_rank}

Rules:
- penalty: emphasize "ranked ahead but missed true target, possible over-marketing risk"
- reward: emphasize "hit true need, recommendation is trustworthy"
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
    if args.hybrid_topk <= 0:
        raise ValueError("--hybrid-topk must be > 0")
    if args.bm25_k1 <= 0:
        raise ValueError("--bm25-k1 must be > 0")
    if not (0.0 <= args.bm25_b <= 1.0):
        raise ValueError("--bm25-b must be in [0, 1]")

    random.seed(args.seed)

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
    client = OpenAI(base_url=api.base_url, api_key=api.api_key, timeout=api.timeout_seconds)
    chat_model = api.default_model

    print("Loading platform catalogs and building retrieval vectors...")
    platform_data = {}
    for idx, pfile in enumerate(platform_files):
        pid = platform_ids[idx]
        items, asin_map, vectors, bm25 = load_platform_items(
            platform_file=pfile,
            max_items=args.max_platform_items,
            emb_dim=args.hash_embedding_dim,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
        )
        if not items:
            raise RuntimeError(f"{pid} has no loaded items: {pfile}")
        platform_data[pid] = {
            "items": items,
            "asin_map": asin_map,
            "vectors": vectors,
            "bm25": bm25,
        }
        print(f"{pid}: loaded_items={len(items)}")

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
        for line_no, line in enumerate(src, start=1):
            if line_no < args.start_line:
                continue
            if args.max_queries and processed >= args.max_queries:
                break

            qrec = json.loads(line)
            if qrec.get("status") != "ok":
                continue

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
                ua_struct = ua_structure_query(
                    client=client,
                    model=chat_model,
                    max_retries=args.max_retries,
                    query_record=qrec,
                )

                candidates: List[PlatformCandidate] = []
                # Pre-decide which platforms (if any) will "hit" the intended item.
                intended_hit_pids: List[str] = []
                intended_exists_any = False
                for pid in platform_ids:
                    asin_map: Dict[str, Item] = platform_data[pid]["asin_map"]
                    if target_asin in asin_map:
                        intended_exists_any = True
                        if random.random() < args.intended_hit_prob:
                            intended_hit_pids.append(pid)

                # If intended item doesn't exist in any platform catalog, skip the whole round.
                if not intended_exists_any:
                    skipped_no_intended += 1
                    round_obj.update(
                        {
                            "status": "skipped_no_intended",
                            "ua_structured_query": ua_struct,
                            "intended_hit_platform_ids": [],
                        }
                    )
                    persist_platform_profiles(profiles_path, platform_profiles)
                    dst.write(json.dumps(round_obj, ensure_ascii=False) + "\n")
                    processed += 1
                    sys.stderr.write(
                        f"\r[run_ua_pa_simulation] {processed}/{total_target} simulated | skipped_no_intended={skipped_no_intended} skipped_intended_not_hit={skipped_intended_not_hit} error_rounds={error_rounds}"
                    )
                    sys.stderr.flush()
                    if args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)
                    continue

                # If no platform hits the intended item, skip the expensive PA top1 retrieval + pitch.
                if not intended_hit_pids:
                    skipped_intended_not_hit += 1
                    round_obj.update(
                        {
                            "status": "skipped_intended_not_hit",
                            "ua_structured_query": ua_struct,
                            "intended_hit_platform_ids": [],
                        }
                    )
                    persist_platform_profiles(profiles_path, platform_profiles)
                    dst.write(json.dumps(round_obj, ensure_ascii=False) + "\n")
                    processed += 1
                    sys.stderr.write(
                        f"\r[run_ua_pa_simulation] {processed}/{total_target} simulated | skipped_no_intended={skipped_no_intended} skipped_intended_not_hit={skipped_intended_not_hit} error_rounds={error_rounds}"
                    )
                    sys.stderr.flush()
                    if args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)
                    continue

                for pid in platform_ids:
                    pdata = platform_data[pid]
                    asin_map: Dict[str, Item] = pdata["asin_map"]
                    items: List[Item] = pdata["items"]
                    vectors: List[List[float]] = pdata["vectors"]
                    bm25: BM25Index = pdata["bm25"]

                    forced_hit = False
                    if pid in intended_hit_pids:
                        chosen = asin_map[target_asin]
                        score = 1.0
                        forced_hit = True
                    else:
                        chosen, score = retrieve_top1(
                            ua_structured=ua_struct,
                            items=items,
                            vectors=vectors,
                            emb_dim=args.hash_embedding_dim,
                            bm25=bm25,
                            hybrid_alpha=args.hybrid_alpha,
                            hybrid_topk=args.hybrid_topk,
                        )

                    style = choose_pitch_style(f"{query_id}|{pid}")
                    pitch = generate_platform_pitch(
                        client=client,
                        model=chat_model,
                        max_retries=args.max_retries,
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
                    c.platform_id for c in candidates if c.item.parent_asin == target_asin
                ]
                if not intended_platforms:
                    skipped_no_intended += 1
                    round_obj.update(
                        {
                            "status": "skipped_no_intended",
                            "ua_structured_query": ua_struct,
                            "platform_candidates": [
                                {
                                    "platform_id": c.platform_id,
                                    "item": c.item.to_pitch_payload(),
                                    "pitch": c.pitch,
                                    "pitch_style": c.pitch_style,
                                    "retrieval_score": c.retrieval_score,
                                    "forced_intended_hit": c.forced_intended_hit,
                                }
                                for c in candidates
                            ],
                        }
                    )
                    persist_platform_profiles(profiles_path, platform_profiles)
                    dst.write(json.dumps(round_obj, ensure_ascii=False) + "\n")
                    processed += 1
                    sys.stderr.write(
                        f"\r[run_ua_pa_simulation] {processed}/{total_target} simulated | skipped_no_intended={skipped_no_intended} skipped_intended_not_hit={skipped_intended_not_hit} error_rounds={error_rounds}"
                    )
                    sys.stderr.flush()
                    if args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)
                    continue

                rank_list, rationale = ua_rank_candidates(
                    client=client,
                    model=chat_model,
                    max_retries=args.max_retries,
                    user_query=user_query,
                    candidates=candidates,
                    platform_profiles=platform_profiles,
                )

                first_target_rank = None
                for idx, pid in enumerate(rank_list, start=1):
                    c = next(x for x in candidates if x.platform_id == pid)
                    if c.item.parent_asin == target_asin:
                        first_target_rank = idx
                        break

                if first_target_rank is None:
                    skipped_no_intended += 1
                    round_obj.update(
                        {
                            "status": "skipped_after_ranking_no_intended",
                            "ua_structured_query": ua_struct,
                            "ua_rank_list": rank_list,
                            "ua_rationale": rationale,
                        }
                    )
                    persist_platform_profiles(profiles_path, platform_profiles)
                    dst.write(json.dumps(round_obj, ensure_ascii=False) + "\n")
                    processed += 1
                    if args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)
                    continue

                reward_pid = rank_list[first_target_rank - 1]
                penalty_pids = rank_list[: first_target_rank - 1]

                for pos, pid in enumerate(rank_list, start=1):
                    cand = next(x for x in candidates if x.platform_id == pid)
                    if pid in penalty_pids:
                        mem = generate_reputation_memory(
                            client=client,
                            model=chat_model,
                            max_retries=args.max_retries,
                            event_type="penalty",
                            platform_id=pid,
                            user_query=user_query,
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
                    elif pid == reward_pid:
                        mem = generate_reputation_memory(
                            client=client,
                            model=chat_model,
                            max_retries=args.max_retries,
                            event_type="reward",
                            platform_id=pid,
                            user_query=user_query,
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
                        "ua_structured_query": ua_struct,
                        "platform_candidates": [
                            {
                                "platform_id": c.platform_id,
                                "item": c.item.to_pitch_payload(),
                                "pitch": c.pitch,
                                "pitch_style": c.pitch_style,
                                "retrieval_score": c.retrieval_score,
                                "forced_intended_hit": c.forced_intended_hit,
                            }
                            for c in candidates
                        ],
                        "intended_platform_ids": intended_platforms,
                        "ua_rank_list": rank_list,
                        "ua_rationale": rationale,
                        "target_rank": first_target_rank,
                        "reward_platform": reward_pid,
                        "penalty_platforms": penalty_pids,
                    }
                )

                persist_platform_profiles(profiles_path, platform_profiles)
                dst.write(json.dumps(round_obj, ensure_ascii=False) + "\n")
                processed += 1

                sys.stderr.write(
                    f"\r[run_ua_pa_simulation] {processed}/{total_target} simulated | skipped_no_intended={skipped_no_intended} skipped_intended_not_hit={skipped_intended_not_hit} error_rounds={error_rounds}"
                )
                sys.stderr.flush()

                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

            except Exception as err:
                error_rounds += 1
                round_obj.update({"status": "error", "error": str(err)})
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
        "hybrid_alpha": args.hybrid_alpha,
        "hybrid_topk": args.hybrid_topk,
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "seed": args.seed,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    run_simulation(args)


if __name__ == "__main__":
    main()
