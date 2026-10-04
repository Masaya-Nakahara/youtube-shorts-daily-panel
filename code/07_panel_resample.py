#!/usr/bin/env python3
"""Panel (cohort) re-sampling for YouTube Shorts VSEO research.

Why
---
`01_collect_pilot.py` is a *coverage* collector: it excludes already-seen video
IDs, so it deliberately never re-observes the same video. That maximises unique
breadth but makes longitudinal (panel) analysis impossible. The most novel
research angles -- early view *velocity*, the Shorts "resurfacing"/second-wind
phenomenon, and subscriber *growth* -- all require tracking the SAME videos and
channels over time.

This script fills that gap. It keeps a cohort ledger of tracked videos and, on
each run, re-fetches their current statistics (views/likes/comments) plus their
channel statistics (subscriberCount/...), appending one observation row per
video to a panel JSONL keyed by (video_id, observed_at).

Cost
----
search.list is NOT used here. Only videos.list (1 unit / 50 ids) and
channels.list (1 unit / 50 ids). Tracking ~2,000 videos costs ~40 + ~40 = ~80
quota units per run -- negligible against the daily 10,000-unit budget, so it
can run every day (ideally more than once early in a video's life).

Commands
--------
  seed      Add videos from a raw JSONL (default: newest data/raw/daily_*.jsonl)
            to the cohort ledger. Use --max-age-days to focus on fresh videos.
  resample  (default) Re-fetch current stats for all active tracked videos +
            their channels and append observations to the panel JSONL.
  status    Print ledger summary (counts, age distribution) and exit.

Files
-----
  state/panel_cohort.json            ledger: per-video tracking metadata
  data/panel/panel_observations.jsonl  append-only panel: one row / observation
  logs/panel_<ts>_meta.json          per-run metadata
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import isodate

ROOT = Path(__file__).resolve().parent.parent


def _load_collector_module():
    """Import helpers from 01_collect_pilot.py (module name starts with a digit,
    so a normal import is not possible -- load it by path)."""
    path = Path(__file__).resolve().parent / "01_collect_pilot.py"
    spec = importlib.util.spec_from_file_location("collect_pilot", path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: dataclasses (Py3.14) resolves field types via
    # sys.modules[cls.__module__], which is None for an unregistered module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Ledger I/O
# ---------------------------------------------------------------------------

def default_ledger_path() -> Path:
    return ROOT / "state" / "panel_cohort.json"


def default_panel_path() -> Path:
    return ROOT / "data" / "panel" / "panel_observations.jsonl"


def load_ledger(path: Path) -> dict:
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data.setdefault("videos", {})
            return data
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning("could not read ledger %s: %s -- starting fresh", path, exc)
    return {"videos": {}, "created_at": datetime.now(timezone.utc).isoformat()}


def save_ledger(path: Path, ledger: dict) -> None:
    ledger["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_dt(s: str):
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def age_days(published_at: str, observed_at: datetime) -> float | None:
    pub = parse_dt(published_at)
    if pub is None:
        return None
    return (observed_at - pub).total_seconds() / 86400.0


def newest_daily_raw() -> Path | None:
    raw_dir = ROOT / "data" / "raw"
    candidates = sorted(raw_dir.glob("daily_*.jsonl"))
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# seed
# ---------------------------------------------------------------------------

def cmd_seed(args, _collector) -> int:
    src = args.source or newest_daily_raw()
    if src is None or not Path(src).is_file():
        logging.error("seed source not found: %s", src)
        return 1
    src = Path(src)
    ledger = load_ledger(args.ledger)
    videos = ledger["videos"]
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    added = 0
    skipped_existing = 0
    skipped_old = 0
    skipped_nodate = 0
    with src.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = str(rec.get("id") or rec.get("video_id") or "").strip()
            if not vid:
                continue
            sn = rec.get("snippet", {}) or {}
            ctx = rec.get("_search_context", {}) or {}
            published_at = sn.get("publishedAt", "")
            a = age_days(published_at, now)
            if a is None:
                skipped_nodate += 1
                continue
            if args.max_age_days is not None and a > args.max_age_days:
                skipped_old += 1
                continue
            if vid in videos:
                skipped_existing += 1
                continue
            videos[vid] = {
                "channel_id": sn.get("channelId", ""),
                "published_at": published_at,
                "first_observed_at": now_iso,
                "last_observed_at": None,
                "n_observations": 0,
                "region": ctx.get("region"),
                "category_id": ctx.get("category_id"),
                "category_name": ctx.get("category_name"),
                "seed_age_days": round(a, 3),
                "seed_source": src.name,
                "retired": False,
                "missing_streak": 0,
            }
            added += 1
            if args.limit and added >= args.limit:
                break

    save_ledger(args.ledger, ledger)
    active = sum(1 for v in videos.values() if not v.get("retired"))
    logging.info(
        "seed done: source=%s added=%d skipped_existing=%d skipped_old=%d "
        "skipped_nodate=%d  ledger_total=%d active=%d",
        src.name, added, skipped_existing, skipped_old, skipped_nodate,
        len(videos), active,
    )
    print(f"added={added} ledger_total={len(videos)} active={active}")
    return 0


# ---------------------------------------------------------------------------
# resample
# ---------------------------------------------------------------------------

def cmd_resample(args, collector) -> int:
    ledger = load_ledger(args.ledger)
    videos = ledger["videos"]
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    run_id = now.strftime("%Y%m%d_%H%M%S")

    active_ids = [
        vid for vid, v in videos.items()
        if not v.get("retired")
        and (
            args.max_track_days is None
            or (age_days(v.get("published_at", ""), now) or 1e9) <= args.max_track_days
        )
    ]
    # Retire videos that have aged out of the tracking window.
    retired_now = 0
    for vid, v in videos.items():
        if v.get("retired"):
            continue
        a = age_days(v.get("published_at", ""), now)
        if args.max_track_days is not None and a is not None and a > args.max_track_days:
            v["retired"] = True
            v["retired_reason"] = "aged_out"
            retired_now += 1

    logging.info(
        "resample start: tracked_total=%d active_in_window=%d aged_out_now=%d dry_run=%s",
        len(videos), len(active_ids), retired_now, args.dry_run,
    )
    if not active_ids:
        logging.info("no active videos to resample")
        print("active=0 (nothing to do)")
        save_ledger(args.ledger, ledger)
        return 0
    if args.dry_run:
        est_quota = (len(active_ids) + 49) // 50
        chan_ids = {videos[v].get("channel_id", "") for v in active_ids}
        est_quota += (len(chan_ids) + 49) // 50
        print(
            f"[dry-run] would resample {len(active_ids)} videos / "
            f"{len([c for c in chan_ids if c])} channels (~{est_quota} quota units)"
        )
        save_ledger(args.ledger, ledger)
        return 0

    # --- live API path ---
    api_key = collector.get_api_key()
    yt = collector.build_youtube_client(api_key)
    quota = collector.QuotaTracker(limit=args.quota_limit)

    details = collector.fetch_video_details(yt, active_ids, quota)
    by_vid = {d.get("id"): d for d in details}
    returned = set(by_vid)
    missing = [v for v in active_ids if v not in returned]

    channel_ids = sorted({
        (by_vid.get(v, {}).get("snippet", {}) or {}).get("channelId")
        or videos[v].get("channel_id", "")
        for v in active_ids
    })
    channel_ids = [c for c in channel_ids if c]
    channel_map = collector.fetch_channel_subs(yt, channel_ids, quota)

    panel_path = args.panel
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with panel_path.open("a", encoding="utf-8") as out:
        for vid in active_ids:
            d = by_vid.get(vid)
            v = videos[vid]
            if d is None:
                # Video disappeared (deleted/private). Count a missing streak.
                v["missing_streak"] = int(v.get("missing_streak", 0)) + 1
                if v["missing_streak"] >= args.retire_after_missing:
                    v["retired"] = True
                    v["retired_reason"] = "missing"
                continue
            v["missing_streak"] = 0
            st = d.get("statistics", {}) or {}
            sn = d.get("snippet", {}) or {}
            cd = d.get("contentDetails", {}) or {}
            status = d.get("status", {}) or {}
            cid = sn.get("channelId") or v.get("channel_id", "")
            ch = channel_map.get(cid, {}) or {}
            ch_st = ch.get("statistics", {}) or {}
            published_at = sn.get("publishedAt") or v.get("published_at", "")
            try:
                dur_s = isodate.parse_duration(cd.get("duration", "PT0S")).total_seconds()
            except Exception:
                dur_s = None

            row = {
                "video_id": vid,
                "channel_id": cid,
                "run_id": run_id,
                "observed_at": now_iso,
                "published_at": published_at,
                "age_days": age_days(published_at, now),
                "views": collector_int(st.get("viewCount")),
                "likes": collector_int(st.get("likeCount")),
                "comments": collector_int(st.get("commentCount")),
                "sub_count": collector_int(ch_st.get("subscriberCount")),
                "ch_total_views": collector_int(ch_st.get("viewCount")),
                "ch_video_count": collector_int(ch_st.get("videoCount")),
                "privacy_status": status.get("privacyStatus"),
                "duration_s": dur_s,
                "region": v.get("region"),
                "category_id": v.get("category_id"),
                "category_name": v.get("category_name"),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            v["last_observed_at"] = now_iso
            v["n_observations"] = int(v.get("n_observations", 0)) + 1

    save_ledger(args.ledger, ledger)

    finished = datetime.now(timezone.utc).isoformat()
    meta = {
        "run_id": run_id,
        "started_at": now_iso,
        "finished_at": finished,
        "active_in_window": len(active_ids),
        "videos_returned": len(returned),
        "videos_missing": len(missing),
        "channels_fetched": len(channel_map),
        "rows_written": written,
        "quota_used": quota.used,
        "quota_limit": args.quota_limit,
        "api_exhausted": quota.api_exhausted,
        "aged_out_now": retired_now,
        "panel_output": str(panel_path),
        "ledger": str(args.ledger),
    }
    meta_path = ROOT / "logs" / f"panel_{run_id}_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info(
        "resample done: rows=%d returned=%d missing=%d quota=%d -> %s",
        written, len(returned), len(missing), quota.used, panel_path,
    )
    print(
        f"rows_written={written} returned={len(returned)} missing={len(missing)} "
        f"quota_used={quota.used} panel={panel_path}"
    )
    return 0


def collector_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def cmd_status(args, _collector) -> int:
    ledger = load_ledger(args.ledger)
    videos = ledger["videos"]
    now = datetime.now(timezone.utc)
    active = [v for v in videos.values() if not v.get("retired")]
    retired = [v for v in videos.values() if v.get("retired")]
    obs_hist: dict[int, int] = {}
    age_buckets = {"<=1d": 0, "<=3d": 0, "<=7d": 0, "<=14d": 0, "<=35d": 0, ">35d": 0}
    for v in active:
        n = int(v.get("n_observations", 0))
        obs_hist[n] = obs_hist.get(n, 0) + 1
        a = age_days(v.get("published_at", ""), now)
        if a is None:
            continue
        for label, hi in [("<=1d", 1), ("<=3d", 3), ("<=7d", 7), ("<=14d", 14), ("<=35d", 35)]:
            if a <= hi:
                age_buckets[label] += 1
                break
        else:
            age_buckets[">35d"] += 1

    print(f"ledger: {args.ledger}")
    print(f"  total tracked : {len(videos)}")
    print(f"  active        : {len(active)}")
    print(f"  retired       : {len(retired)}")
    print(f"  observations/video (active): {dict(sorted(obs_hist.items()))}")
    print(f"  age distribution (active)  : {age_buckets}")
    panel = args.panel
    if panel.is_file():
        nrows = sum(1 for _ in panel.open(encoding="utf-8"))
        print(f"  panel rows    : {nrows}  ({panel})")
    else:
        print(f"  panel rows    : 0  (no file yet: {panel})")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", default="resample",
                   choices=["seed", "resample", "status"],
                   help="seed | resample (default) | status")
    p.add_argument("--source", type=Path, default=None,
                   help="[seed] raw JSONL to seed from (default: newest data/raw/daily_*.jsonl)")
    p.add_argument("--max-age-days", type=float, default=14.0,
                   help="[seed] only track videos whose publish age <= this (default 14; set 0 or negative to disable)")
    p.add_argument("--limit", type=int, default=None,
                   help="[seed] cap number of new videos added this run")
    p.add_argument("--max-track-days", type=float, default=35.0,
                   help="[resample] stop tracking videos older than this many days since publish")
    p.add_argument("--retire-after-missing", type=int, default=3,
                   help="[resample] retire a video after this many consecutive missing observations")
    p.add_argument("--quota-limit", type=int, default=None,
                   help="[resample] stop before exceeding this quota-unit budget")
    p.add_argument("--dry-run", action="store_true",
                   help="[resample] do not call the API; just report what would be fetched")
    p.add_argument("--ledger", type=Path, default=default_ledger_path(),
                   help="cohort ledger JSON path")
    p.add_argument("--panel", type=Path, default=default_panel_path(),
                   help="append-only panel observations JSONL path")
    args = p.parse_args()

    if args.max_age_days is not None and args.max_age_days <= 0:
        args.max_age_days = None

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    collector = _load_collector_module()
    if args.command == "seed":
        return cmd_seed(args, collector)
    if args.command == "status":
        return cmd_status(args, collector)
    return cmd_resample(args, collector)


if __name__ == "__main__":
    sys.exit(main())
