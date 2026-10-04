#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""30_probe_missing_reason.py -- Classify why panel videos became unavailable.

For every video the panel retired with retired_reason == "missing" (plus a random
control sample of still-active videos), fetch the public watch page and read the
player's playabilityStatus / reason text. No API quota is used. Results let the
paper distinguish uploader removal, private, terminated account, Community
Guidelines / copyright removals, and false "missing" (video still available).

Usage:
  python scripts/30_probe_missing_reason.py [--controls 150] [--delay 1.0] [--limit N]
Outputs:
  data/probe/missing_reasons_<ts>.jsonl   one line per probed video (resumable)
  data/probe/missing_reasons_<ts>_summary.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
RX_STATUS = re.compile(r'"playabilityStatus":\{"status":"([A-Z_]+)"')
RX_REASON = re.compile(r'"reason":"((?:[^"\\]|\\.)*)"')
RX_SUBREASON = re.compile(r'"subreason":\{"simpleText":"((?:[^"\\]|\\.)*)"')
RX_TITLE = re.compile(r'"videoDetails":\{"videoId":"([A-Za-z0-9_-]{11})"')


def classify(status: str | None, reason: str, http: int) -> str:
    r = (reason or "").lower()
    if status == "OK":
        return "available"
    if "removed by the uploader" in r:
        return "uploader_removed"
    if "private" in r:
        return "private"
    if "terminated" in r or "account associated" in r:
        return "account_terminated"
    if "community guidelines" in r or "terms of service" in r or "violating" in r:
        return "policy_removed"
    if "copyright" in r:
        return "copyright"
    if "not available in your country" in r or "country" in r:
        return "geo_blocked"
    if "age" in r and "sign in" in r:
        return "age_restricted"
    if status in ("LOGIN_REQUIRED",):
        return "login_required"
    if "unavailable" in r or status in ("ERROR", "UNPLAYABLE"):
        return "unavailable_generic"
    if http != 200:
        return f"http_{http}"
    return "unknown"


def probe(session: requests.Session, vid: str, timeout: float = 20.0) -> dict:
    url = f"https://www.youtube.com/watch?v={vid}&hl=en"
    try:
        resp = session.get(url, timeout=timeout, allow_redirects=True)
    except requests.RequestException as e:
        return {"video_id": vid, "http": -1, "status": None, "reason": f"EXC:{type(e).__name__}", "cls": "fetch_error"}
    html = resp.text
    m = RX_STATUS.search(html)
    status = m.group(1) if m else None
    reason = ""
    if m:
        tail = html[m.end(): m.end() + 2500]
        # collect every human-readable string in the playability block (reason, errorScreen title/description, subreason)
        texts = []
        for t in re.findall(r'"(?:reason|content|simpleText|text)":"((?:[^"\\]|\\.)*)"', tail):
            t = t.encode("utf-8").decode("unicode_escape", errors="ignore").strip()
            if t and t not in texts and not t.startswith(("CA", "Eg", "Q0")) and len(t) < 200:
                texts.append(t)
        reason = " | ".join(texts)
    consent = "consent.youtube.com" in resp.url
    out = {"video_id": vid, "http": resp.status_code, "status": status, "reason": reason[:300], "consent_redirect": consent,
           "has_video_details": bool(RX_TITLE.search(html)), "cls": classify(status, reason, resp.status_code)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path, default=ROOT / "state/panel_cohort.json")
    ap.add_argument("--controls", type=int, default=150)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", type=Path, default=None, help="existing jsonl to continue")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = ROOT / "data/probe"; outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.resume or (outdir / f"missing_reasons_{ts}.jsonl")
    done = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            try: done.add(json.loads(line)["video_id"])
            except Exception: pass
    led = pd.DataFrame.from_dict(json.load(open(args.ledger, encoding="utf-8"))["videos"], orient="index")
    miss = led[led.retired_reason == "missing"]
    active = led[led.retired != True]
    rnd = random.Random(args.seed)
    controls = rnd.sample(list(active.index), min(args.controls, len(active)))
    targets = [(v, "missing") for v in miss.index] + [(v, "control") for v in controls]
    if args.limit: targets = targets[: args.limit]
    todo = [(v, g) for v, g in targets if v not in done]
    print(f"targets={len(targets)} done={len(done)} todo={len(todo)} out={out_path}", file=sys.stderr, flush=True)
    s = requests.Session(); s.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9,ja;q=0.8"})
    s.cookies.set("CONSENT", "YES+cb", domain=".youtube.com"); s.cookies.set("SOCS", "CAI", domain=".youtube.com")
    n = 0; t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for vid, grp in todo:
            rec = probe(s, vid); rec["group"] = grp; rec["seed_age_days"] = float(led.loc[vid, "seed_age_days"]) if vid in led.index else None
            rec["category_name"] = led.loc[vid, "category_name"] if vid in led.index else None
            rec["channel_id"] = led.loc[vid, "channel_id"] if vid in led.index else None
            rec["fetched_at"] = datetime.now(timezone.utc).isoformat()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            n += 1
            if rec["http"] == 429:
                print("429 rate limited; sleeping 120 s", file=sys.stderr, flush=True); time.sleep(120)
            if n % 100 == 0:
                print(f"  {n}/{len(todo)} elapsed={time.time()-t0:.0f}s", file=sys.stderr, flush=True)
            time.sleep(args.delay + rnd.random() * 0.5)
    rows = [json.loads(l) for l in open(out_path, encoding="utf-8")]
    df = pd.DataFrame(rows)
    summ = {"n": int(len(df)), "by_group_cls": df.groupby(["group", "cls"]).size().unstack(fill_value=0).to_dict("index"),
            "missing_cls_share": (df[df.group == "missing"].cls.value_counts(normalize=True).round(4)).to_dict(),
            "missing_by_category_cls": df[df.group == "missing"].groupby(["category_name", "cls"]).size().unstack(fill_value=0).to_dict("index"),
            "reason_texts_top": df[df.group == "missing"].reason.value_counts().head(15).to_dict(), "generated_at": ts, "out": str(out_path)}
    sp = Path(str(out_path).replace(".jsonl", "_summary.json"))
    sp.write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ["missing_cls_share"], ensure_ascii=False)); print("summary:", sp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
