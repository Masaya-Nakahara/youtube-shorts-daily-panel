#!/usr/bin/env python3
"""Pilot data collection for YouTube Shorts VSEO research.

Strategy
--------
- Loop over (region, category) combinations.
- For each, run a search for "#shorts" hashtag, biased to short videos.
- Fetch full video details + channel subscriber counts.
- Filter post-hoc to actual Shorts (duration <= 60s).
- Write everything as JSONL to data/raw/pilot_<timestamp>.jsonl.

Quota usage (default settings)
------------------------------
- search.list = 100 units/call (50 results max)
- videos.list = 1 unit/call (up to 50 ids)
- channels.list = 1 unit/call (up to 50 ids)
- 12 categories x 2 regions = 24 search calls -> ~2400 units
- Plus details/channel calls -> ~50 units
- Total budget: ~2500 units (well within daily 10k limit).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import isodate
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# Stable, well-populated YouTube category IDs.
# Source: https://developers.google.com/youtube/v3/docs/videoCategories
CATEGORIES = {
    1: "Film & Animation",
    2: "Autos & Vehicles",
    10: "Music",
    15: "Pets & Animals",
    17: "Sports",
    19: "Travel & Events",
    20: "Gaming",
    22: "People & Blogs",
    23: "Comedy",
    24: "Entertainment",
    25: "News & Politics",
    26: "Howto & Style",
    27: "Education",
    28: "Science & Technology",
}

REGIONS = [
    ("JP", "ja"),
    ("US", "en"),
    # India (regionCode=IN) for the symmetric 3rd segment. Two language pairs
    # (Hindi + English) cover India's bilingual short-form space. Only activated
    # by passing --regions IN; the daily UI offers JP/US only, so the daily
    # pipeline is unaffected.
    ("IN", "hi"),
    ("IN", "en"),
]


@dataclass
class QuotaTracker:
    used: int = 0
    limit: int | None = None
    api_exhausted: bool = False

    def can_spend(self, cost: int) -> bool:
        return not self.api_exhausted and (
            self.limit is None or self.used + cost <= self.limit
        )

    def add(self, cost: int, op: str) -> None:
        self.used += cost
        logging.info("  quota +%d (%s)  -> total=%d", cost, op, self.used)

    def mark_api_exhausted(self) -> None:
        self.api_exhausted = True
        logging.warning("YouTube API quota is exhausted; stopping further API calls")


def is_quota_exceeded(error: HttpError) -> bool:
    details = getattr(error, "error_details", None) or []
    return any(
        isinstance(item, dict) and item.get("reason") == "quotaExceeded"
        for item in details
    )


def log_http_error(action: str, error: HttpError) -> None:
    details = getattr(error, "error_details", None) or []
    reasons = [
        item.get("reason", "")
        for item in details
        if isinstance(item, dict) and item.get("reason")
    ]
    logging.warning(
        "%s failed: HTTP %s reasons=%s",
        action,
        getattr(error.resp, "status", "?"),
        ",".join(reasons) if reasons else "unknown",
    )


def get_api_key() -> str:
    """Read API key from env, fall back to ~/.config/youtube_api/key."""
    key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if key:
        return key
    fallback = Path.home() / ".config" / "youtube_api" / "key"
    if fallback.is_file():
        return fallback.read_text(encoding="utf-8").strip()
    raise RuntimeError(
        "YOUTUBE_API_KEY not found in env or ~/.config/youtube_api/key"
    )


def build_youtube_client(api_key: str):
    return build(
        "youtube",
        "v3",
        developerKey=api_key,
        cache_discovery=False,
    )


def extract_video_id(record: dict) -> str:
    video_id = record.get("id") or record.get("video_id")
    if isinstance(video_id, str):
        return video_id.strip()
    if isinstance(video_id, dict):
        return str(video_id.get("videoId") or "").strip()
    return ""


def read_jsonl_video_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    try:
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    video_id = extract_video_id(json.loads(line))
                except json.JSONDecodeError:
                    logging.warning("skip malformed JSONL line: %s:%d", path, line_no)
                    continue
                if video_id:
                    ids.add(video_id)
    except OSError as exc:
        logging.warning("could not read existing raw file %s: %s", path, exc)
    return ids


def load_existing_video_ids(
    raw_dir: Path,
    exclude_paths: set[Path] | None = None,
) -> tuple[set[str], int]:
    if not raw_dir.is_dir():
        return set(), 0
    excluded = {p.resolve() for p in (exclude_paths or set())}
    ids: set[str] = set()
    files_scanned = 0
    for path in sorted(raw_dir.glob("*.jsonl")):
        if path.resolve() in excluded:
            continue
        ids.update(read_jsonl_video_ids(path))
        files_scanned += 1
    return ids, files_scanned


def search_shorts(
    yt,
    region: str,
    relevance_lang: str,
    category_id: int,
    max_results: int,
    order: str,
    quota: QuotaTracker,
    page_token: str | None = None,
) -> tuple[list[str], str | None]:
    """Return up to max_results videoIds matching #shorts hashtag in
    a specific category and region."""
    if not quota.can_spend(100):
        logging.warning(
            "skip search region=%s cat=%d order=%s: quota limit would be exceeded",
            region,
            category_id,
            order,
        )
        return [], None
    page_label = "next" if page_token else "first"
    quota.add(
        100,
        f"search region={region} cat={category_id} order={order} page={page_label}",
    )
    params = {
        "part": "id",
        "q": "#shorts",
        "type": "video",
        "videoCategoryId": str(category_id),
        "regionCode": region,
        "relevanceLanguage": relevance_lang,
        "videoDuration": "short",  # under 4 minutes
        "maxResults": min(50, max(1, max_results)),
        "order": order,
        "safeSearch": "none",
    }
    if page_token:
        params["pageToken"] = page_token
    try:
        resp = yt.search().list(**params).execute()
    except HttpError as e:
        log_http_error(f"search region={region} cat={category_id} order={order}", e)
        if is_quota_exceeded(e):
            quota.mark_api_exhausted()
        return [], None
    ids = [
        it["id"]["videoId"]
        for it in resp.get("items", [])
        if it.get("id", {}).get("kind") == "youtube#video"
    ]
    return ids, resp.get("nextPageToken")


def fetch_video_details(yt, ids: list[str], quota: QuotaTracker) -> list[dict]:
    """Fetch full video details for up to 50 ids per call."""
    out: list[dict] = []
    for i in range(0, len(ids), 50):
        if not quota.can_spend(1):
            logging.warning("skip videos.list: quota limit would be exceeded")
            break
        chunk = ids[i : i + 50]
        quota.add(1, f"videos.list({len(chunk)})")
        try:
            resp = (
                yt.videos()
                .list(
                    part="snippet,statistics,contentDetails,topicDetails,status",
                    id=",".join(chunk),
                    maxResults=50,
                )
                .execute()
            )
        except HttpError as e:
            log_http_error("videos.list", e)
            if is_quota_exceeded(e):
                quota.mark_api_exhausted()
                break
            continue
        out.extend(resp.get("items", []))
    return out


def fetch_channel_subs(yt, channel_ids: Iterable[str], quota: QuotaTracker) -> dict[str, dict]:
    """Fetch subscriber count and channel metadata for a set of channels."""
    out: dict[str, dict] = {}
    cids = sorted(set(channel_ids))
    for i in range(0, len(cids), 50):
        if not quota.can_spend(1):
            logging.warning("skip channels.list: quota limit would be exceeded")
            break
        chunk = cids[i : i + 50]
        quota.add(1, f"channels.list({len(chunk)})")
        try:
            resp = (
                yt.channels()
                .list(
                    part="snippet,statistics,topicDetails",
                    id=",".join(chunk),
                    maxResults=50,
                )
                .execute()
            )
        except HttpError as e:
            log_http_error("channels.list", e)
            if is_quota_exceeded(e):
                quota.mark_api_exhausted()
                break
            continue
        for it in resp.get("items", []):
            out[it["id"]] = it
    return out


def is_actual_short(item: dict, max_seconds: int = 60) -> bool:
    """Filter to videos with duration <= max_seconds.
    Note: vertical/portrait detection requires looking at recordingDetails
    which is not available in the public API. We rely on duration only,
    plus the #shorts hashtag and videoDuration=short used in search."""
    dur_iso = item.get("contentDetails", {}).get("duration", "PT0S")
    try:
        seconds = isodate.parse_duration(dur_iso).total_seconds()
    except Exception:
        return False
    return 0 < seconds <= max_seconds


def collect(
    out_path: Path,
    max_per_combo: int,
    log_path: Path,
    orders: list[str],
    regions: list[str],
    categories: list[int],
    quota_limit: int | None,
    meta_out: Path | None,
    exclude_existing: bool,
    existing_raw_dir: Path | None,
    max_search_pages: int,
) -> None:
    max_per_combo = max(1, min(50, max_per_combo))
    max_search_pages = max(1, max_search_pages)
    api_key = get_api_key()
    yt = build_youtube_client(api_key)
    quota = QuotaTracker(limit=quota_limit)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )

    started_at = datetime.now(timezone.utc).isoformat()
    logging.info("START pilot collection at %s", started_at)
    logging.info(
        (
            "output=%s max_per_combo=%d orders=%s regions=%s categories=%s "
            "quota_limit=%s exclude_existing=%s max_search_pages=%d"
        ),
        out_path,
        max_per_combo,
        ",".join(orders),
        ",".join(regions),
        ",".join(str(c) for c in categories),
        quota_limit,
        exclude_existing,
        max_search_pages,
    )

    n_kept = 0
    n_dropped = 0
    search_pages_used = 0
    search_ids_returned = 0
    skipped_existing_ids = 0
    skipped_run_duplicate_ids = 0
    skipped_page_duplicate_ids = 0
    all_records: list[dict] = []
    channel_ids_seen: set[str] = set()
    video_ids_seen: set[str] = set()
    existing_raw_dir = existing_raw_dir or out_path.parent
    existing_video_ids: set[str] = set()
    existing_files_scanned = 0
    if exclude_existing:
        existing_video_ids, existing_files_scanned = load_existing_video_ids(
            existing_raw_dir,
            exclude_paths={out_path},
        )
    logging.info(
        "existing exclusion enabled=%s raw_dir=%s files=%d loaded_video_ids=%d",
        exclude_existing,
        existing_raw_dir,
        existing_files_scanned,
        len(existing_video_ids),
    )

    selected_regions = [(r, lang) for r, lang in REGIONS if r in set(regions)]
    selected_categories = {
        cat_id: cat_name
        for cat_id, cat_name in CATEGORIES.items()
        if cat_id in set(categories)
    }

    for region, lang in selected_regions:
        if quota.api_exhausted:
            break
        for cat_id, cat_name in selected_categories.items():
            if quota.api_exhausted:
                break
            for order in orders:
                if quota.api_exhausted:
                    break
                logging.info(
                    "region=%s lang=%s cat=%d (%s) order=%s",
                    region,
                    lang,
                    cat_id,
                    cat_name,
                    order,
                )
                ids: list[str] = []
                next_page_token: str | None = None
                combo_pages = 0
                combo_existing_skips = 0
                combo_run_duplicate_skips = 0
                for _ in range(max_search_pages):
                    used_before = quota.used
                    page_ids, next_page_token = search_shorts(
                        yt,
                        region,
                        lang,
                        cat_id,
                        50,
                        order,
                        quota,
                        page_token=next_page_token,
                    )
                    if quota.used > used_before:
                        search_pages_used += 1
                        combo_pages += 1
                    if not page_ids:
                        break
                    search_ids_returned += len(page_ids)
                    unique_page_ids = list(dict.fromkeys(page_ids))
                    skipped_page_duplicate_ids += len(page_ids) - len(unique_page_ids)
                    for vid in unique_page_ids:
                        if exclude_existing and vid in existing_video_ids:
                            skipped_existing_ids += 1
                            combo_existing_skips += 1
                            continue
                        if vid in video_ids_seen:
                            skipped_run_duplicate_ids += 1
                            combo_run_duplicate_skips += 1
                            continue
                        ids.append(vid)
                        video_ids_seen.add(vid)
                        if len(ids) >= max_per_combo:
                            break
                    if len(ids) >= max_per_combo or not next_page_token:
                        break
                    if quota.api_exhausted:
                        break
                logging.info(
                    (
                        "selected_new_ids=%d search_pages=%d skipped_existing=%d "
                        "skipped_run_duplicates=%d"
                    ),
                    len(ids),
                    combo_pages,
                    combo_existing_skips,
                    combo_run_duplicate_skips,
                )
                if not ids:
                    continue
                details = fetch_video_details(yt, ids, quota)
                if quota.api_exhausted and not details:
                    break
                for item in details:
                    if not is_actual_short(item):
                        n_dropped += 1
                        continue
                    # tag with our search context
                    item["_search_context"] = {
                        "region": region,
                        "relevance_language": lang,
                        "category_id": cat_id,
                        "category_name": cat_name,
                        "order": order,
                    }
                    all_records.append(item)
                    channel_ids_seen.add(item.get("snippet", {}).get("channelId", ""))
                    n_kept += 1

    # Fetch channel info for all referenced channels.
    logging.info("fetching channel info for %d unique channels", len(channel_ids_seen))
    channel_map = fetch_channel_subs(
        yt, [c for c in channel_ids_seen if c], quota
    )
    logging.info("fetched channel info for %d channels", len(channel_map))

    # Attach channel info to records.
    for rec in all_records:
        cid = rec.get("snippet", {}).get("channelId", "")
        if cid in channel_map:
            rec["_channel"] = channel_map[cid]

    # Write JSONL.
    with out_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    finished_at = datetime.now(timezone.utc).isoformat()
    records_with_channel = sum(1 for rec in all_records if rec.get("_channel"))
    selected_candidate_total = (
        len(video_ids_seen) + skipped_existing_ids + skipped_run_duplicate_ids
    )
    selected_new_candidate_rate = (
        len(video_ids_seen) / selected_candidate_total
        if selected_candidate_total
        else 0.0
    )
    logging.info(
        (
            "DONE finished_at=%s kept=%d dropped=%d quota_used=%d "
            "api_exhausted=%s records_with_channel=%d existing_loaded=%d "
            "search_pages=%d skipped_existing=%d selected_new_candidate_rate=%.4f"
        ),
        finished_at,
        n_kept,
        n_dropped,
        quota.used,
        quota.api_exhausted,
        records_with_channel,
        len(existing_video_ids),
        search_pages_used,
        skipped_existing_ids,
        selected_new_candidate_rate,
    )
    logging.info("output written: %s", out_path)
    if meta_out is not None:
        meta_out.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "started_at": started_at,
            "finished_at": finished_at,
            "kept": n_kept,
            "dropped": n_dropped,
            "quota_used": quota.used,
            "quota_limit": quota_limit,
            "api_exhausted": quota.api_exhausted,
            "records_with_channel": records_with_channel,
            "unique_channels": len(channel_ids_seen),
            "channels_fetched": len(channel_map),
            "exclude_existing": exclude_existing,
            "existing_raw_dir": str(existing_raw_dir),
            "existing_raw_files_scanned": existing_files_scanned,
            "existing_video_ids_loaded": len(existing_video_ids),
            "max_search_pages": max_search_pages,
            "search_pages_used": search_pages_used,
            "search_ids_returned": search_ids_returned,
            "selected_new_ids": len(video_ids_seen),
            "skipped_existing_ids": skipped_existing_ids,
            "skipped_run_duplicate_ids": skipped_run_duplicate_ids,
            "skipped_page_duplicate_ids": skipped_page_duplicate_ids,
            "selected_new_candidate_rate": selected_new_candidate_rate,
            "output": str(out_path),
            "log": str(log_path),
            "orders": orders,
            "regions": regions,
            "categories": categories,
        }
        with meta_out.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logging.info("metadata written: %s", meta_out)


def parse_category_args(values: list[str] | None) -> list[int]:
    if not values or any(v.lower() == "all" for v in values):
        return sorted(CATEGORIES)
    categories: list[int] = []
    for value in values:
        try:
            cat_id = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid category id: {value}") from exc
        if cat_id not in CATEGORIES:
            raise argparse.ArgumentTypeError(f"unknown category id: {cat_id}")
        categories.append(cat_id)
    return sorted(set(categories))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL path. Default: data/raw/pilot_<timestamp>.jsonl",
    )
    p.add_argument(
        "--max-per-combo",
        type=int,
        default=50,
        help="Max video ids to fetch per (region, category) combo (1-50).",
    )
    p.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Log file path. Default: logs/collect_<timestamp>.log",
    )
    p.add_argument(
        "--orders",
        nargs="+",
        default=["relevance"],
        choices=["relevance", "date", "viewCount"],
        help="Search orders to rotate. Multiple orders improve sampling diversity.",
    )
    p.add_argument(
        "--regions",
        nargs="+",
        default=["JP", "US"],
        choices=[r for r, _ in REGIONS],
        help="Region codes to collect.",
    )
    p.add_argument(
        "--categories",
        nargs="+",
        default=["all"],
        help="Category IDs to collect, or 'all'.",
    )
    p.add_argument(
        "--quota-limit",
        type=int,
        default=None,
        help="Stop before this run would exceed the given quota-unit budget.",
    )
    p.add_argument(
        "--max-search-pages",
        type=int,
        default=3,
        help=(
            "Max search result pages per (region, category, order). "
            "Each extra page costs 100 quota units."
        ),
    )
    p.add_argument(
        "--existing-raw-dir",
        type=Path,
        default=None,
        help="Directory containing previous JSONL files to avoid collecting again.",
    )
    p.add_argument(
        "--exclude-existing",
        dest="exclude_existing",
        action="store_true",
        default=True,
        help="Skip video IDs already present in existing raw JSONL files.",
    )
    p.add_argument(
        "--include-existing",
        dest="exclude_existing",
        action="store_false",
        help="Allow videos that are already present in existing raw JSONL files.",
    )
    p.add_argument(
        "--meta-out",
        type=Path,
        default=None,
        help="Optional JSON metadata output with actual quota usage.",
    )
    args = p.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or (project_root / "data" / "raw" / f"pilot_{ts}.jsonl")
    log_path = args.log or (project_root / "logs" / f"collect_{ts}.log")

    try:
        collect(
            out_path,
            max_per_combo=args.max_per_combo,
            log_path=log_path,
            orders=args.orders,
            regions=args.regions,
            categories=parse_category_args(args.categories),
            quota_limit=args.quota_limit,
            meta_out=args.meta_out,
            exclude_existing=args.exclude_existing,
            existing_raw_dir=args.existing_raw_dir,
            max_search_pages=args.max_search_pages,
        )
    except Exception as e:
        logging.exception("collection failed: %s", e)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
