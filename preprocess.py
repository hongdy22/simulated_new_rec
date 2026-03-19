#!/usr/bin/env python3
import argparse
import json
import random
import sqlite3
from pathlib import Path
from typing import Dict, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge Amazon meta/review JSONL by parent_asin and split items into platform "
            "catalogs with controlled overlap."
        )
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=Path("/mnt/d/桌面/new_rec/dataset/meta_Musical_Instruments.jsonl"),
        help="Path to meta_Musical_Instruments.jsonl",
    )
    parser.add_argument(
        "--review-path",
        type=Path,
        default=Path("/mnt/d/桌面/new_rec/dataset/Musical_Instruments.jsonl"),
        help="Path to Musical_Instruments.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory for merged file and platform catalogs",
    )
    parser.add_argument(
        "--platform-count",
        type=int,
        default=5,
        help="Number of platform catalogs to create",
    )
    parser.add_argument(
        "--overlap-prob",
        type=float,
        default=0.2,
        help="Probability that an item is additionally copied to each non-primary platform",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible platform assignment",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("output/reviews_tmp.sqlite"),
        help="Temporary SQLite path used for review indexing",
    )
    parser.add_argument(
        "--drop-tmp-db",
        action="store_true",
        help="Delete temporary SQLite db after successful generation",
    )
    parser.add_argument(
        "--min-history-len",
        type=int,
        default=5,
        help="Minimum number of interactions required per user history",
    )
    parser.add_argument(
        "--max-history-len",
        type=int,
        default=20,
        help="Maximum number of interactions kept per user history (keep latest)",
    )
    return parser.parse_args()


def collect_meta_asins(meta_path: Path) -> Set[str]:
    asins: Set[str] = set()
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            asin = obj.get("parent_asin")
            if asin:
                asins.add(asin)
    return asins


def build_review_index(review_path: Path, valid_asins: Set[str], db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode = WAL")
    cur.execute("PRAGMA synchronous = NORMAL")
    cur.execute(
        "CREATE TABLE reviews ("
        "parent_asin TEXT NOT NULL, "
        "user_id TEXT, "
        "timestamp INTEGER, "
        "rating REAL, "
        "text TEXT"
        ")"
    )

    batch: List[tuple] = []
    batch_size = 10000
    with review_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            asin = obj.get("parent_asin")
            if not asin or asin not in valid_asins:
                continue
            batch.append(
                (
                    asin,
                    obj.get("user_id"),
                    obj.get("timestamp"),
                    obj.get("rating"),
                    obj.get("text"),
                )
            )
            if len(batch) >= batch_size:
                cur.executemany("INSERT INTO reviews VALUES (?, ?, ?, ?, ?)", batch)
                conn.commit()
                batch.clear()

    if batch:
        cur.executemany("INSERT INTO reviews VALUES (?, ?, ?, ?, ?)", batch)
        conn.commit()

    cur.execute("CREATE INDEX idx_reviews_parent_asin ON reviews(parent_asin)")
    cur.execute("CREATE INDEX idx_reviews_user_time ON reviews(user_id, timestamp)")
    conn.commit()
    return conn


def load_reviews(conn: sqlite3.Connection, parent_asin: str) -> List[Dict]:
    rows = conn.execute(
        "SELECT rating, text FROM reviews WHERE parent_asin = ?", (parent_asin,)
    ).fetchall()
    return [{"rating": row[0], "text": row[1]} for row in rows]


def collect_asin_title_map(meta_path: Path) -> Dict[str, str]:
    asin_title: Dict[str, str] = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            asin = obj.get("parent_asin")
            if not asin:
                continue
            asin_title[asin] = obj.get("title")
    return asin_title


def generate_user_histories(
    conn: sqlite3.Connection,
    output_dir: Path,
    asin_title_map: Dict[str, str],
    min_history_len: int,
    max_history_len: int,
) -> Tuple[Path, int, int]:
    user_history_path = output_dir / "user_histories.jsonl"
    user_count = 0
    interaction_count = 0

    def flush_user_record(fp, user_id: str, interactions: List[Dict]) -> Tuple[int, int]:
        if len(interactions) < min_history_len:
            return 0, 0

        trimmed = interactions[-max_history_len:]
        record = {
            "user_id": user_id,
            "interactions": trimmed,
        }
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        return 1, len(trimmed)

    with user_history_path.open("w", encoding="utf-8") as f:
        cur = conn.execute(
            "SELECT user_id, parent_asin, timestamp, rating, text "
            "FROM reviews "
            "WHERE user_id IS NOT NULL "
            "ORDER BY user_id, timestamp"
        )

        current_user_id = None
        current_interactions: List[Dict] = []

        for row in cur:
            user_id, parent_asin, timestamp, rating, text = row
            if current_user_id is None:
                current_user_id = user_id

            if user_id != current_user_id:
                uc, ic = flush_user_record(f, current_user_id, current_interactions)
                user_count += uc
                interaction_count += ic

                current_user_id = user_id
                current_interactions = []

            current_interactions.append(
                {
                    "parent_asin": parent_asin,
                    "title": asin_title_map.get(parent_asin),
                    "timestamp": timestamp,
                    "rating": rating,
                    "text": text,
                }
            )

        if current_user_id is not None:
            uc, ic = flush_user_record(f, current_user_id, current_interactions)
            user_count += uc
            interaction_count += ic

    return user_history_path, user_count, interaction_count


def choose_platforms(
    rng: random.Random, platform_count: int, overlap_prob: float
) -> List[int]:
    primary = rng.randrange(platform_count)
    selected = [primary]
    for platform_id in range(platform_count):
        if platform_id == primary:
            continue
        if rng.random() < overlap_prob:
            selected.append(platform_id)
    return selected


def generate_outputs(
    meta_path: Path,
    output_dir: Path,
    conn: sqlite3.Connection,
    platform_count: int,
    overlap_prob: float,
    seed: int,
    min_history_len: int,
    max_history_len: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    platforms_dir = output_dir / "platforms"
    platforms_dir.mkdir(parents=True, exist_ok=True)

    merged_path = output_dir / "all_items_with_reviews.jsonl"
    platform_paths = [
        platforms_dir / f"platform_{i + 1}.jsonl" for i in range(platform_count)
    ]

    rng = random.Random(seed)

    platform_counts = [0 for _ in range(platform_count)]
    total_items = 0

    with merged_path.open("w", encoding="utf-8") as merged_fp:
        platform_fps = [p.open("w", encoding="utf-8") for p in platform_paths]
        try:
            with meta_path.open("r", encoding="utf-8") as meta_fp:
                for line in meta_fp:
                    meta_obj = json.loads(line)
                    parent_asin = meta_obj.get("parent_asin")
                    if not parent_asin:
                        continue

                    record = {
                        "parent_asin": parent_asin,
                        "main_category": meta_obj.get("main_category"),
                        "title": meta_obj.get("title"),
                        "average_rating": meta_obj.get("average_rating"),
                        "rating_number": meta_obj.get("rating_number"),
                        "description": meta_obj.get("description"),
                        "reviews": load_reviews(conn, parent_asin),
                    }

                    line_out = json.dumps(record, ensure_ascii=False)
                    merged_fp.write(line_out + "\n")

                    selected_platforms = choose_platforms(
                        rng=rng,
                        platform_count=platform_count,
                        overlap_prob=overlap_prob,
                    )
                    for pid in selected_platforms:
                        platform_fps[pid].write(line_out + "\n")
                        platform_counts[pid] += 1

                    total_items += 1
        finally:
            for fp in platform_fps:
                fp.close()

    asin_title_map = collect_asin_title_map(meta_path)
    user_history_path, user_count, interaction_count = generate_user_histories(
        conn=conn,
        output_dir=output_dir,
        asin_title_map=asin_title_map,
        min_history_len=min_history_len,
        max_history_len=max_history_len,
    )

    summary = {
        "total_meta_items": total_items,
        "total_users": user_count,
        "total_user_interactions": interaction_count,
        "platform_count": platform_count,
        "overlap_prob": overlap_prob,
        "seed": seed,
        "min_history_len": min_history_len,
        "max_history_len": max_history_len,
        "merged_file": str(merged_path),
        "user_history_file": str(user_history_path),
        "platform_files": [str(p) for p in platform_paths],
        "items_per_platform": {
            f"platform_{i + 1}": platform_counts[i] for i in range(platform_count)
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    if args.platform_count <= 0:
        raise ValueError("--platform-count must be > 0")
    if not (0.0 <= args.overlap_prob <= 1.0):
        raise ValueError("--overlap-prob must be in [0, 1]")
    if args.min_history_len <= 0:
        raise ValueError("--min-history-len must be > 0")
    if args.max_history_len < args.min_history_len:
        raise ValueError("--max-history-len must be >= --min-history-len")

    meta_asins = collect_meta_asins(args.meta_path)
    conn = build_review_index(args.review_path, meta_asins, args.db_path)
    try:
        generate_outputs(
            meta_path=args.meta_path,
            output_dir=args.output_dir,
            conn=conn,
            platform_count=args.platform_count,
            overlap_prob=args.overlap_prob,
            seed=args.seed,
            min_history_len=args.min_history_len,
            max_history_len=args.max_history_len,
        )
    finally:
        conn.close()

    if args.drop_tmp_db and args.db_path.exists():
        args.db_path.unlink()


if __name__ == "__main__":
    main()