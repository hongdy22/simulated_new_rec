#!/usr/bin/env python3
import argparse
import html
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PLATFORM_IDS = ["P1", "P2", "P3", "P4", "P5"]
PLATFORM_PERSONALITIES = {
    "P1": "honest",
    "P2": "expert",
    "P3": "concise",
    "P4": "friendly",
    "P5": "promo",
}
DEFAULT_ROUNDS_PATH = Path(__file__).resolve().parent / "output" / "sim" / "simulation_rounds.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize simulation_rounds.jsonl as a self-contained HTML report."
    )
    parser.add_argument(
        "rounds_path",
        nargs="?",
        type=Path,
        default=DEFAULT_ROUNDS_PATH,
        help="Path to simulation_rounds.jsonl",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Optional output HTML path. Defaults to <rounds_dir>/simulation_rounds_report.html",
    )
    return parser.parse_args()


def load_rounds(rounds_path: Path) -> List[Dict]:
    if not rounds_path.exists():
        raise FileNotFoundError(f"Rounds file not found: {rounds_path}")
    rounds: List[Dict] = []
    with rounds_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rounds.append(json.loads(line))
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON at line {line_no} in {rounds_path}: {err}") from err
    if not rounds:
        raise ValueError(f"No records found in {rounds_path}")
    return rounds


def safe_pct(numer: float, denom: float) -> float:
    if not denom:
        return 0.0
    return numer / denom


def avg(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def style_of(round_obj: Dict) -> str:
    ua = round_obj.get("ua_structured_query") or {}
    return str(ua.get("style") or "unknown")


def purchased_platform_of(round_obj: Dict) -> str:
    return str(round_obj.get("purchased_platform") or round_obj.get("reward_platform") or "")


def error_bucket(message: str) -> str:
    msg = str(message or "").strip()
    if not msg:
        return "Unknown error"
    lowered = msg.lower()
    if "401" in lowered or "令牌验证失败" in msg:
        return "401 auth/token failure"
    if "429" in lowered:
        return "429 rate limit"
    if "timeout" in lowered:
        return "timeout"
    if "empty llm response" in lowered:
        return "empty llm response"
    if len(msg) > 120:
        return msg[:117] + "..."
    return msg


def candidate_hit_reason(round_obj: Dict, candidate: Dict) -> str:
    reason = str(candidate.get("intended_hit_reason") or "").strip().lower()
    if reason in {"forced", "retrieved", "not_intended"}:
        return reason

    target_asin = str(round_obj.get("target_asin") or "")
    item = candidate.get("item") or {}
    item_asin = str(item.get("parent_asin") or "")
    if not target_asin or item_asin != target_asin:
        return "not_intended"
    if bool(candidate.get("forced_intended_hit")):
        return "forced"
    return "retrieved"


def compute_metrics(rounds: List[Dict]) -> Dict:
    total_rounds = len(rounds)
    settled_rounds = [r for r in rounds if r.get("status") == "settled"]
    error_rounds = [r for r in rounds if r.get("status") == "error"]

    target_rank_values = [
        int(r.get("target_rank"))
        for r in settled_rounds
        if isinstance(r.get("target_rank"), int) or str(r.get("target_rank") or "").isdigit()
    ]
    top1_hits = sum(1 for r in settled_rounds if int(r.get("target_rank") or 0) == 1)
    target_rank_missing_count = 0
    profile_update_count = 0

    style_totals: Counter = Counter()
    purchase_by_style_platform: Dict[str, Counter] = defaultdict(Counter)
    round_hit_source_counts: Counter = Counter()
    rounds_with_forced_hit = 0
    rounds_with_retrieved_hit = 0

    platform_metrics: Dict[str, Dict[str, float]] = {
        pid: {
            "purchase_count": 0,
            "top_rank_count": 0,
            "intended_presence_count": 0,
            "forced_hit_count": 0,
            "retrieved_hit_count": 0,
        }
        for pid in PLATFORM_IDS
    }

    error_buckets: Counter = Counter()
    error_examples: List[Tuple[str, str]] = []

    for round_obj in rounds:
        style = style_of(round_obj)
        status = str(round_obj.get("status") or "unknown")
        style_totals[style] += 1

        if status == "error":
            bucket = error_bucket(str(round_obj.get("error") or ""))
            error_buckets[bucket] += 1
            if len(error_examples) < 8:
                error_examples.append((str(round_obj.get("query_id") or ""), str(round_obj.get("error") or "")))

        round_has_forced_hit = False
        round_has_retrieved_hit = False
        for candidate in round_obj.get("platform_candidates") or []:
            pid = str(candidate.get("platform_id") or "")
            if pid not in platform_metrics:
                continue

            hit_reason = candidate_hit_reason(round_obj, candidate)
            if hit_reason == "forced":
                platform_metrics[pid]["forced_hit_count"] += 1
                round_has_forced_hit = True
            elif hit_reason == "retrieved":
                platform_metrics[pid]["retrieved_hit_count"] += 1
                round_has_retrieved_hit = True

        if round_has_forced_hit:
            rounds_with_forced_hit += 1
            round_hit_source_counts["forced"] += 1
        if round_has_retrieved_hit:
            rounds_with_retrieved_hit += 1
            round_hit_source_counts["retrieved"] += 1

        if status != "settled":
            continue

        if str(round_obj.get("profile_memory_update") or "").strip():
            profile_update_count += 1

        target_rank = int(round_obj.get("target_rank") or 0)
        if target_rank <= 0:
            target_rank_missing_count += 1

        purchased_platform = purchased_platform_of(round_obj)
        if purchased_platform:
            purchase_by_style_platform[style][purchased_platform] += 1
            if purchased_platform in platform_metrics:
                platform_metrics[purchased_platform]["purchase_count"] += 1

        ua_rank_list = round_obj.get("ua_rank_list") or []
        if ua_rank_list:
            top_pid = str(ua_rank_list[0])
            if top_pid in platform_metrics:
                platform_metrics[top_pid]["top_rank_count"] += 1

        intended_pids = round_obj.get("intended_platform_ids") or []
        intended_set = {str(pid) for pid in intended_pids}
        for pid in intended_set:
            if pid in platform_metrics:
                platform_metrics[pid]["intended_presence_count"] += 1

    platform_rows: List[Dict] = []
    settled_count = len(settled_rounds)
    for pid in PLATFORM_IDS:
        metrics = platform_metrics[pid]
        platform_rows.append(
            {
                "platform_id": pid,
                "purchase_count": int(metrics["purchase_count"]),
                "purchase_rate": safe_pct(metrics["purchase_count"], settled_count),
                "top_rank_count": int(metrics["top_rank_count"]),
                "top_rank_rate": safe_pct(metrics["top_rank_count"], settled_count),
                "intended_presence_count": int(metrics["intended_presence_count"]),
                "intended_presence_rate": safe_pct(metrics["intended_presence_count"], settled_count),
                "forced_hit_count": int(metrics["forced_hit_count"]),
                "retrieved_hit_count": int(metrics["retrieved_hit_count"]),
            }
        )

    purchase_style_matrix: List[Dict] = []
    for style in sorted(style_totals):
        row = {"style": style}
        row.update({pid: int(purchase_by_style_platform[style][pid]) for pid in PLATFORM_IDS})
        purchase_style_matrix.append(row)

    target_rank_counts = Counter(target_rank_values)

    return {
        "total_rounds": total_rounds,
        "settled_count": len(settled_rounds),
        "error_count": len(error_rounds),
        "top1_hits": top1_hits,
        "top1_rate_on_settled": safe_pct(top1_hits, len(settled_rounds)),
        "avg_target_rank": avg(target_rank_values),
        "target_rank_missing_count": target_rank_missing_count,
        "profile_update_count": profile_update_count,
        "round_hit_source_counts": dict(round_hit_source_counts),
        "rounds_with_forced_hit": rounds_with_forced_hit,
        "rounds_with_retrieved_hit": rounds_with_retrieved_hit,
        "platform_rows": platform_rows,
        "purchase_style_matrix": purchase_style_matrix,
        "target_rank_counts": dict(sorted(target_rank_counts.items())),
        "error_buckets": dict(error_buckets.most_common()),
        "error_examples": error_examples,
    }


def pct_text(value: float) -> str:
    return f"{value * 100:.1f}%"


def num_text(value: float) -> str:
    if abs(value - int(value)) < 1e-9:
        return str(int(value))
    return f"{value:.2f}"


def esc(text: object) -> str:
    return html.escape(str(text))


def bar_rows(title: str, rows: List[Tuple[str, float, str]], color: str = "#2f6fed") -> str:
    if not rows:
        return f"<section><h2>{esc(title)}</h2><p>No data.</p></section>"
    parts = [f"<section><h2>{esc(title)}</h2><div class='bar-chart'>"]
    for label, frac, value_text in rows:
        width = max(0.0, min(100.0, frac * 100.0))
        parts.append(
            "<div class='bar-row'>"
            f"<div class='bar-label'>{esc(label)}</div>"
            "<div class='bar-track'>"
            f"<div class='bar-fill' style='width:{width:.1f}%;background:{color};'></div>"
            "</div>"
            f"<div class='bar-value'>{esc(value_text)}</div>"
            "</div>"
        )
    parts.append("</div></section>")
    return "".join(parts)


def short_text(text: object, limit: int = 280) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def chip_list(values: List[object], class_name: str = "") -> str:
    cleaned = [str(v).strip() for v in values if str(v).strip()]
    if not cleaned:
        return "<span class='empty-note'>None</span>"
    chip_class = "chip"
    if class_name:
        chip_class += f" {class_name}"
    return "".join(f"<span class='{chip_class}'>{esc(v)}</span>" for v in cleaned)


def status_badge(status: object) -> str:
    value = str(status or "unknown").strip() or "unknown"
    status_class = "status-unknown"
    if value == "settled":
        status_class = "status-settled"
    elif value.startswith("skipped"):
        status_class = "status-skipped"
    elif value == "error":
        status_class = "status-error"
    return f"<span class='status-badge {status_class}'>{esc(value)}</span>"


def reason_badge(reason: str) -> str:
    label = {
        "forced": "forced hit",
        "retrieved": "retrieved hit",
        "not_intended": "not intended",
    }.get(reason, reason or "unknown")
    class_name = "reason-neutral"
    if reason == "forced":
        class_name = "reason-forced"
    elif reason == "retrieved":
        class_name = "reason-retrieved"
    return f"<span class='reason-badge {class_name}'>{esc(label)}</span>"


def render_rank_list(round_obj: Dict) -> str:
    rank_list = [str(pid) for pid in (round_obj.get("ua_rank_list") or []) if str(pid).strip()]
    if not rank_list:
        return "<p class='small'>No ranking output for this case.</p>"

    target_rank = int(round_obj.get("target_rank") or 0) if str(round_obj.get("target_rank") or "").isdigit() else 0
    purchased_platform = purchased_platform_of(round_obj)
    penalty_set = {str(pid) for pid in (round_obj.get("penalty_platforms") or [])}
    candidate_map = {
        str(candidate.get("platform_id") or ""): candidate
        for candidate in (round_obj.get("platform_candidates") or [])
    }

    parts = ["<div class='rank-list'>"]
    for pos, pid in enumerate(rank_list, start=1):
        candidate = candidate_map.get(pid, {})
        title = str((candidate.get("item") or {}).get("title") or "")
        chips = []
        if pid == purchased_platform:
            chips.append("<span class='chip chip-good'>purchased</span>")
        if pid in penalty_set:
            chips.append("<span class='chip chip-warn'>penalty</span>")
        if target_rank == pos:
            chips.append("<span class='chip chip-accent'>target rank</span>")
        reason = candidate_hit_reason(round_obj, candidate) if candidate else "not_intended"
        if reason in {"forced", "retrieved"}:
            chips.append(reason_badge(reason))
        parts.append(
            "<div class='rank-item'>"
            f"<div class='rank-pos'>#{pos}</div>"
            "<div class='rank-main'>"
            f"<div class='rank-head'><strong>{esc(pid)}</strong> {''.join(chips)}</div>"
            f"<div class='small'>{esc(short_text(title, 110) or 'No candidate title')}</div>"
            "</div>"
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def render_candidate_cards(round_obj: Dict) -> str:
    candidates = round_obj.get("platform_candidates") or []
    if not candidates:
        return "<p class='small'>No platform candidate details recorded for this case.</p>"

    cards = ["<div class='candidate-grid'>"]
    for candidate in candidates:
        platform_id = str(candidate.get("platform_id") or "")
        item = candidate.get("item") or {}
        title = str(item.get("title") or "")
        asin = str(item.get("parent_asin") or "")
        score = candidate.get("retrieval_score")
        score_text = num_text(float(score)) if isinstance(score, (int, float)) else "N/A"
        reason = candidate_hit_reason(round_obj, candidate)
        reviews = item.get("review_snippets") or []
        review_text = " | ".join(short_text(x, 120) for x in reviews[:2] if str(x).strip())
        cards.append(
            "<article class='candidate-card'>"
            "<div class='candidate-top'>"
            f"<div><strong>{esc(platform_id)}</strong></div>"
            f"<div>{reason_badge(reason)}</div>"
            "</div>"
            f"<h3>{esc(short_text(title, 140) or 'Untitled item')}</h3>"
            "<div class='kv-list'>"
            f"<div><span>ASIN</span><strong>{esc(asin or 'N/A')}</strong></div>"
            f"<div><span>Score</span><strong>{esc(score_text)}</strong></div>"
            f"<div><span>Pitch Style</span><strong>{esc(candidate.get('pitch_style') or 'N/A')}</strong></div>"
            f"<div><span>Category</span><strong>{esc(short_text(item.get('main_category') or 'N/A', 50))}</strong></div>"
            "</div>"
            f"<p class='candidate-pitch'>{esc(short_text(candidate.get('pitch') or '', 420) or 'No pitch')}</p>"
            "<details>"
            "<summary>More item detail</summary>"
            f"<p class='small'><strong>Description:</strong> {esc(short_text(item.get('description') or '', 600) or 'N/A')}</p>"
            f"<p class='small'><strong>Reviews:</strong> {esc(review_text or 'N/A')}</p>"
            "</details>"
            "</article>"
        )
    cards.append("</div>")
    return "".join(cards)


def render_case_pages(rounds: List[Dict]) -> Tuple[str, str, str]:
    page_buttons: List[str] = []
    select_options: List[str] = []
    pages: List[str] = []
    total = len(rounds)

    for idx, round_obj in enumerate(rounds, start=1):
        status = str(round_obj.get("status") or "unknown")
        style = style_of(round_obj)
        ua = round_obj.get("ua_structured_query") or {}
        query_text = round_obj.get("query_text") or ua.get("query_rewrite") or ua.get("user_need") or ""
        target_item = round_obj.get("target_item") or {}
        intended_pids = [str(pid) for pid in (round_obj.get("intended_platform_ids") or [])]
        penalty_pids = [str(pid) for pid in (round_obj.get("penalty_platforms") or [])]
        forced_pids = [
            str(candidate.get("platform_id") or "")
            for candidate in (round_obj.get("platform_candidates") or [])
            if candidate_hit_reason(round_obj, candidate) == "forced"
        ]
        retrieved_pids = [
            str(candidate.get("platform_id") or "")
            for candidate in (round_obj.get("platform_candidates") or [])
            if candidate_hit_reason(round_obj, candidate) == "retrieved"
        ]
        purchased_pid = purchased_platform_of(round_obj)
        purchased_item = round_obj.get("purchased_item") or {}
        profile_memory = str(round_obj.get("profile_memory_update") or "")
        ua_rationale = str(round_obj.get("ua_rationale") or "")

        page_buttons.append(
            f"<button type='button' class='page-btn' data-case-target='{idx - 1}'>{idx}</button>"
        )
        select_options.append(
            f"<option value='{idx - 1}'>Case {idx}: {esc(status)} | {esc(short_text(query_text, 48) or str(round_obj.get('query_id') or ''))}</option>"
        )

        summary_cards = (
            "<div class='case-summary-cards'>"
            f"<div class='mini-card'><div class='label'>Status</div><div>{status_badge(status)}</div></div>"
            f"<div class='mini-card'><div class='label'>Style</div><div>{esc(style)}</div></div>"
            f"<div class='mini-card'><div class='label'>Target ASIN</div><div><code>{esc(round_obj.get('target_asin') or 'N/A')}</code></div></div>"
            f"<div class='mini-card'><div class='label'>Purchased Platform</div><div>{esc(purchased_pid or 'N/A')}</div></div>"
            f"<div class='mini-card'><div class='label'>Target Rank</div><div>{esc(round_obj.get('target_rank') or 'N/A')}</div></div>"
            f"<div class='mini-card'><div class='label'>Query ID</div><div><code>{esc(round_obj.get('query_id') or 'N/A')}</code></div></div>"
            "</div>"
        )

        pages.append(
            f"<section class='case-page' data-case-index='{idx - 1}'>"
            "<div class='case-header'>"
            f"<div><h2>Case {idx}</h2><p class='small'>Line {esc(round_obj.get('source_line') or 'N/A')} | User <code>{esc(round_obj.get('user_id') or 'N/A')}</code></p></div>"
            f"<div>{status_badge(status)}</div>"
            "</div>"
            f"{summary_cards}"
            "<div class='grid two' style='margin-top: 16px;'>"
            "<section>"
            "<h3>Query</h3>"
            f"<p><strong>Original Query:</strong> {esc(query_text or 'N/A')}</p>"
            f"<p><strong>User Need:</strong> {esc(ua.get('user_need') or 'N/A')}</p>"
            f"<p><strong>Query Rewrite:</strong> {esc(ua.get('query_rewrite') or query_text or 'N/A')}</p>"
            f"<p><strong>Keywords:</strong> {chip_list(ua.get('keywords') or [])}</p>"
            f"<p><strong>Constraints:</strong> {chip_list(ua.get('constraints') or [])}</p>"
            f"<p><strong>Real Next Item:</strong> {esc(target_item.get('title') or 'N/A')}</p>"
            f"<p><strong>Real Next Review:</strong> {esc(short_text(target_item.get('user_review') or '', 200) or 'N/A')}</p>"
            "</section>"
            "<section>"
            "<h3>Outcome</h3>"
            f"<p><strong>Intended Platforms:</strong> {chip_list(intended_pids, 'chip-accent')}</p>"
            f"<p><strong>Forced Hit Platforms:</strong> {chip_list(forced_pids, 'chip-warn')}</p>"
            f"<p><strong>Retrieved Hit Platforms:</strong> {chip_list(retrieved_pids, 'chip-good')}</p>"
            f"<p><strong>Purchased Platform:</strong> {esc(purchased_pid or 'N/A')}</p>"
            f"<p><strong>Purchased Item:</strong> {esc(short_text(purchased_item.get('title') or 'N/A', 140))}</p>"
            f"<p><strong>Profile Update:</strong> {esc(profile_memory or 'None')}</p>"
            f"<p><strong>Penalty Platforms:</strong> {chip_list(penalty_pids, 'chip-warn')}</p>"
            f"<p><strong>Error:</strong> {esc(short_text(round_obj.get('error') or 'None', 300))}</p>"
            "</section>"
            "</div>"
            "<section style='margin-top: 16px;'>"
            "<h3>UA Rank List</h3>"
            f"{render_rank_list(round_obj)}"
            "</section>"
            "<section style='margin-top: 16px;'>"
            "<h3>UA Rationale</h3>"
            f"<p>{esc(short_text(ua_rationale or 'N/A', 900))}</p>"
            "</section>"
            "<section style='margin-top: 16px;'>"
            "<h3>Platform Candidates</h3>"
            f"{render_candidate_cards(round_obj)}"
            "</section>"
            "</section>"
        )

    return "".join(page_buttons), "".join(select_options), "".join(pages)


def render_report(rounds_path: Path, metrics: Dict, rounds: List[Dict]) -> str:
    input_name = rounds_path.name
    round_hit_source_counts = metrics["round_hit_source_counts"]
    case_page_buttons, case_select_options, case_pages_html = render_case_pages(rounds)

    target_rank_rows = [
        (f"target_rank={rank}", safe_pct(count, metrics["settled_count"]), f"{count}")
        for rank, count in metrics["target_rank_counts"].items()
    ]

    hit_source_rows = [
        (
            f"{label} rounds",
            safe_pct(count, metrics["settled_count"]),
            f"{count} ({pct_text(safe_pct(count, metrics['settled_count']))})",
        )
        for label, count in [
            ("forced", round_hit_source_counts.get("forced", 0)),
            ("retrieved", round_hit_source_counts.get("retrieved", 0)),
        ]
        if count > 0
    ]

    platform_table_rows = []
    for row in metrics["platform_rows"]:
        platform_label = row["platform_id"]
        personality = PLATFORM_PERSONALITIES.get(platform_label)
        if personality:
            platform_label = f"{platform_label} ({personality})"
        platform_table_rows.append(
            "<tr>"
            f"<td>{esc(platform_label)}</td>"
            f"<td>{row['purchase_count']}</td>"
            f"<td>{pct_text(row['purchase_rate'])}</td>"
            f"<td>{row['intended_presence_count']}</td>"
            f"<td>{pct_text(row['intended_presence_rate'])}</td>"
            f"<td>{row['forced_hit_count']}</td>"
            f"<td>{row['retrieved_hit_count']}</td>"
            "</tr>"
        )

    reward_matrix_rows = []
    for row in metrics["purchase_style_matrix"]:
        cells = [f"<td>{esc(row['style'])}</td>"]
        max_value = max(row[pid] for pid in PLATFORM_IDS) if PLATFORM_IDS else 0
        for pid in PLATFORM_IDS:
            value = row[pid]
            alpha = 0.12
            if max_value > 0:
                alpha = 0.12 + 0.40 * (value / max_value)
            cells.append(
                f"<td style='background: rgba(47,111,237,{alpha:.3f});'>{value}</td>"
            )
        reward_matrix_rows.append("<tr>" + "".join(cells) + "</tr>")

    error_bucket_rows = []
    for bucket, count in metrics["error_buckets"].items():
        error_bucket_rows.append(f"<tr><td>{esc(bucket)}</td><td>{count}</td></tr>")
    if not error_bucket_rows:
        error_bucket_rows.append("<tr><td colspan='2'>No errors.</td></tr>")

    error_example_rows = []
    for query_id, message in metrics["error_examples"]:
        error_example_rows.append(
            "<tr>"
            f"<td>{esc(query_id)}</td>"
            f"<td class='left'>{esc(message)}</td>"
            "</tr>"
        )
    if not error_example_rows:
        error_example_rows.append("<tr><td colspan='2'>No error examples.</td></tr>")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    title = f"Simulation Report - {rounds_path.name}"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #1e2430;
      --muted: #6f7787;
      --line: #d9d2c5;
      --accent: #2f6fed;
      --accent-2: #d96f32;
      --good: #2d8a5f;
      --warn: #c77f1a;
      --bad: #bf3d3d;
      --shadow: 0 18px 36px rgba(33, 39, 51, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Georgia", "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(47, 111, 237, 0.08), transparent 28%),
        radial-gradient(circle at top right, rgba(217, 111, 50, 0.10), transparent 26%),
        linear-gradient(180deg, #faf7f1 0%, var(--bg) 100%);
    }}
    .page {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    header {{
      margin-bottom: 24px;
      padding: 24px;
      background: linear-gradient(135deg, rgba(47,111,237,0.10), rgba(217,111,50,0.10));
      border: 1px solid rgba(47,111,237,0.14);
      border-radius: 22px;
      box-shadow: var(--shadow);
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    h1 {{
      font-size: 34px;
      line-height: 1.1;
      letter-spacing: 0.01em;
    }}
    h2 {{
      font-size: 24px;
      margin-bottom: 14px;
    }}
    p, li {{ color: var(--muted); line-height: 1.5; }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 10px 18px;
      font-size: 15px;
      margin-top: 14px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin: 22px 0 26px;
    }}
    .card, section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
    }}
    .card {{
      padding: 18px;
    }}
    .card .label {{
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .card .value {{
      font-size: 34px;
      font-weight: 700;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
    }}
    @media (min-width: 980px) {{
      .grid.two {{
        grid-template-columns: 1fr 1fr;
      }}
    }}
    section {{
      padding: 20px;
    }}
    .bar-chart {{
      display: grid;
      gap: 12px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 160px 1fr 110px;
      gap: 12px;
      align-items: center;
    }}
    .bar-label, .bar-value {{
      font-size: 14px;
    }}
    .bar-track {{
      height: 14px;
      background: #ebe4d8;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 999px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      vertical-align: top;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      color: var(--muted);
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    .left {{
      text-align: left;
      word-break: break-word;
    }}
    .footnote {{
      margin-top: 14px;
      font-size: 13px;
      color: var(--muted);
    }}
    .small {{
      font-size: 13px;
    }}
    code {{
      background: rgba(47,111,237,0.08);
      padding: 1px 6px;
      border-radius: 8px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    .empty-note {{
      color: var(--muted);
      font-style: italic;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 3px 9px;
      margin: 2px 6px 2px 0;
      border-radius: 999px;
      background: rgba(47,111,237,0.08);
      border: 1px solid rgba(47,111,237,0.12);
      font-size: 12px;
      color: var(--ink);
    }}
    .chip-good {{
      background: rgba(45,138,95,0.10);
      border-color: rgba(45,138,95,0.20);
    }}
    .chip-warn {{
      background: rgba(199,127,26,0.10);
      border-color: rgba(199,127,26,0.22);
    }}
    .chip-accent {{
      background: rgba(47,111,237,0.10);
      border-color: rgba(47,111,237,0.20);
    }}
    .status-badge, .reason-badge {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      border: 1px solid transparent;
    }}
    .status-settled {{
      background: rgba(45,138,95,0.12);
      color: var(--good);
      border-color: rgba(45,138,95,0.18);
    }}
    .status-skipped {{
      background: rgba(199,127,26,0.12);
      color: var(--warn);
      border-color: rgba(199,127,26,0.20);
    }}
    .status-error {{
      background: rgba(191,61,61,0.12);
      color: var(--bad);
      border-color: rgba(191,61,61,0.18);
    }}
    .status-unknown, .reason-neutral {{
      background: rgba(111,119,135,0.10);
      color: var(--muted);
      border-color: rgba(111,119,135,0.16);
    }}
    .reason-forced {{
      background: rgba(199,127,26,0.12);
      color: var(--warn);
      border-color: rgba(199,127,26,0.22);
    }}
    .reason-retrieved {{
      background: rgba(45,138,95,0.12);
      color: var(--good);
      border-color: rgba(45,138,95,0.20);
    }}
    .case-controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 14px;
    }}
    .page-btn, .nav-btn, .case-select {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 12px;
      padding: 8px 12px;
      font-size: 14px;
    }}
    .page-btn, .nav-btn {{
      cursor: pointer;
    }}
    .page-btn.active {{
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }}
    .page-links {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 16px;
    }}
    .case-indicator {{
      font-weight: 700;
      color: var(--ink);
    }}
    .case-page {{
      display: none;
      margin-top: 10px;
    }}
    .case-page.active {{
      display: block;
    }}
    .case-header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
      margin-bottom: 14px;
    }}
    .case-summary-cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }}
    .mini-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      background: rgba(255,255,255,0.60);
    }}
    .mini-card .label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .rank-list {{
      display: grid;
      gap: 10px;
    }}
    .rank-item {{
      display: grid;
      grid-template-columns: 52px 1fr;
      gap: 12px;
      align-items: center;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255,255,255,0.64);
    }}
    .rank-pos {{
      font-size: 24px;
      font-weight: 700;
      color: var(--accent);
      text-align: center;
    }}
    .rank-head {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-bottom: 4px;
    }}
    .candidate-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
    }}
    .candidate-card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      background: rgba(255,255,255,0.68);
    }}
    .candidate-card h3 {{
      font-size: 18px;
      line-height: 1.25;
      margin-bottom: 10px;
    }}
    .candidate-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 10px;
    }}
    .kv-list {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px 12px;
      margin-bottom: 12px;
    }}
    .kv-list div {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 13px;
    }}
    .kv-list span {{
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-size: 11px;
    }}
    .candidate-pitch {{
      color: var(--ink);
      margin: 0 0 10px;
    }}
    details summary {{
      cursor: pointer;
      color: var(--accent);
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <div class="page">
    <header>
        <h1>{esc(title)}</h1>
        <p>Round-level simulation dashboard for ranking outcomes, target-hit diagnostics, and failure patterns.</p>
        <div class="meta">
        <div><strong>Input</strong><br><code>{esc(input_name)}</code></div>
        <div><strong>Generated</strong><br>{esc(generated_at)}</div>
        <div><strong>Settled Rounds</strong><br>{metrics['settled_count']}</div>
        <div><strong>Error Rounds</strong><br>{metrics['error_count']}</div>
      </div>
    </header>

    <div class="cards">
      <div class="card"><div class="label">Total Rounds</div><div class="value">{metrics['total_rounds']}</div></div>
      <div class="card"><div class="label">Top-1 Hit Rate</div><div class="value">{pct_text(metrics['top1_rate_on_settled'])}</div></div>
      <div class="card"><div class="label">Avg Target Rank (Observed)</div><div class="value">{num_text(metrics['avg_target_rank'])}</div></div>
      <div class="card"><div class="label">Target Missing In Rank List</div><div class="value">{metrics['target_rank_missing_count']}</div></div>
      <div class="card"><div class="label">Forced Hit Rounds</div><div class="value">{metrics['rounds_with_forced_hit']}</div></div>
      <div class="card"><div class="label">Retrieved Hit Rounds (No Force)</div><div class="value">{metrics['rounds_with_retrieved_hit']}</div></div>
    </div>

    <div class="grid two" style="margin-top: 18px;">
      {bar_rows("Observed Target Rank Distribution", target_rank_rows, color="#2d8a5f")}
      {bar_rows("Rounds With Target Hit Source", hit_source_rows, color="#8a4fd1")}
    </div>

    <div class="grid two" style="margin-top: 18px;">
      <section>
        <h2>Definitions</h2>
        <p class="small"><strong>Top-1 hit rate</strong> means <code>target_rank == 1</code> among settled rounds, when the real next item appeared in ranked candidates.</p>
        <p class="small"><strong>Observed target rank</strong> is diagnostic only. Settlement always follows the UA's rank-1 platform as the purchased outcome.</p>
        <p class="small"><strong>Target Missing In Rank List</strong> counts settled rounds where the real next item never appeared anywhere in the final ranked candidates.</p>
        <p class="small"><strong>Forced Hit Rounds</strong> counts settled rounds where at least one platform got the target item through the forced-hit override.</p>
        <p class="small"><strong>Retrieved Hit Rounds (No Force)</strong> counts settled rounds where at least one platform naturally retrieved the target item without that override.</p>
        <p class="small"><strong>Rounds With Target Hit Source</strong> is counted by round, not by item. If a round contains both forced and natural hits, it contributes to both bars.</p>
        <p class="small"><strong>Purchase rate</strong> means how often a platform became the final purchased platform among settled rounds.</p>
        <p class="small"><strong>Intended presence rate</strong> means how often a platform's candidate item matched the real next item among settled rounds.</p>
      </section>
      <section>
        <h2>Reading Hint</h2>
        <p class="small">If <code>Forced Hit Rounds</code> is much larger than <code>Retrieved Hit Rounds (No Force)</code>, target coverage is still relying mostly on injection rather than natural retrieval.</p>
        <p class="small">If <code>Retrieved Hit Rounds (No Force)</code> stays at zero, the retriever is not surfacing the real next item on its own.</p>
        <p class="small">A platform can still win the purchase even when the target item is missing, because settlement now always follows the UA's top-ranked result.</p>
      </section>
    </div>

    <section style="margin-top: 18px;">
      <h2>Platform Performance</h2>
      <table>
        <thead>
          <tr>
            <th>Platform</th>
            <th>Purchased Count</th>
            <th>Purchase Rate</th>
            <th>Intended Present</th>
            <th>Intended Rate</th>
            <th>Forced Hits</th>
            <th>Retrieved Hits</th>
          </tr>
        </thead>
        <tbody>
          {''.join(platform_table_rows)}
        </tbody>
      </table>
    </section>

    <section style="margin-top: 18px;">
      <h2>Purchased Platform By Query Style</h2>
      <table>
        <thead>
          <tr>
            <th>Style</th>
            {''.join(f'<th>{pid}</th>' for pid in PLATFORM_IDS)}
          </tr>
        </thead>
        <tbody>
          {''.join(reward_matrix_rows)}
        </tbody>
      </table>
    </section>

    <section id="case-details" style="margin-top: 18px;">
      <h2>Case Details</h2>
      <p class="small">Keep the overview at the top, then use the controls below to flip through individual cases one by one.</p>
      <div class="case-controls">
        <button type="button" class="nav-btn" id="case-prev">Previous</button>
        <span class="case-indicator" id="case-indicator">Case 1 / {len(rounds)}</span>
        <button type="button" class="nav-btn" id="case-next">Next</button>
        <select id="case-select" class="case-select">
          {case_select_options}
        </select>
      </div>
      <div class="page-links" id="case-page-links">
        {case_page_buttons}
      </div>
      <div id="case-pages">
        {case_pages_html}
      </div>
    </section>

    <div class="grid two" style="margin-top: 18px;">
      <section>
        <h2>Error Categories</h2>
        <table>
          <thead>
            <tr><th>Category</th><th>Count</th></tr>
          </thead>
          <tbody>
            {''.join(error_bucket_rows)}
          </tbody>
        </table>
      </section>
      <section>
        <h2>Error Examples</h2>
        <table>
          <thead>
            <tr><th>Query ID</th><th>Error</th></tr>
          </thead>
          <tbody>
            {''.join(error_example_rows)}
          </tbody>
        </table>
      </section>
    </div>
  </div>
  <script>
    (function() {{
      const pages = Array.from(document.querySelectorAll('.case-page'));
      const buttons = Array.from(document.querySelectorAll('.page-btn'));
      const select = document.getElementById('case-select');
      const indicator = document.getElementById('case-indicator');
      const prev = document.getElementById('case-prev');
      const next = document.getElementById('case-next');
      if (!pages.length) {{
        return;
      }}

      function normalizeIndex(index) {{
        if (index < 0) return 0;
        if (index >= pages.length) return pages.length - 1;
        return index;
      }}

      function showCase(index) {{
        const active = normalizeIndex(index);
        pages.forEach((page, i) => {{
          page.classList.toggle('active', i === active);
        }});
        buttons.forEach((btn, i) => {{
          btn.classList.toggle('active', i === active);
        }});
        if (select) {{
          select.value = String(active);
        }}
        if (indicator) {{
          indicator.textContent = 'Case ' + (active + 1) + ' / ' + pages.length;
        }}
        if (prev) {{
          prev.disabled = active === 0;
        }}
        if (next) {{
          next.disabled = active === pages.length - 1;
        }}
      }}

      buttons.forEach((btn) => {{
        btn.addEventListener('click', () => {{
          showCase(Number(btn.dataset.caseTarget || 0));
        }});
      }});
      if (select) {{
        select.addEventListener('change', () => {{
          showCase(Number(select.value || 0));
        }});
      }}
      if (prev) {{
        prev.addEventListener('click', () => {{
          const current = Number(select ? select.value : 0);
          showCase(current - 1);
        }});
      }}
      if (next) {{
        next.addEventListener('click', () => {{
          const current = Number(select ? select.value : 0);
          showCase(current + 1);
        }});
      }}
      showCase(0);
    }})();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    rounds_path = args.rounds_path.resolve()
    output_html = args.output_html.resolve() if args.output_html else rounds_path.with_name("simulation_rounds_report.html")

    rounds = load_rounds(rounds_path)
    metrics = compute_metrics(rounds)
    report_html = render_report(rounds_path, metrics, rounds)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(report_html, encoding="utf-8")
    print(json.dumps({"rounds_path": str(rounds_path), "output_html": str(output_html), "rounds": len(rounds)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
