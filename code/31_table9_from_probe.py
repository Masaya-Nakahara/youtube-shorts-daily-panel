#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""31_table9_from_probe.py -- Aggregate 30_probe_missing_reason.py output into Table 9 of the panel paper.

Rows: uploader_removed / no_reason (generic "unavailable" without videoDetails) / terminated / policy /
copyright / private / exists (videoDetails present).  Columns: videos first observed <=24 h after
publication and later retired as missing; the subset that had passed 35 days since publication at the
last panel run; shares by creator location (JP/US/IN); control sample.

Usage:
  python scripts/31_table9_from_probe.py [--probe data/probe/missing_reasons_<ts>.jsonl]
Output:
  <probe stem>_table9.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("m28", ROOT / "scripts/28_panel_lifecycle.py")
m28 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m28)

ROWS = ["uploader_removed", "no_reason", "terminated", "policy", "copyright", "private", "exists"]


def table(d: pd.DataFrame) -> dict:
    ex = d.has_video_details.astype(bool)
    c = {"uploader_removed": int((d.cls == "uploader_removed").sum()),
         "no_reason": int(((d.cls == "unavailable_generic") & ~ex).sum()),
         "terminated": int((d.cls == "account_terminated").sum()),
         "policy": int((d.cls == "policy_removed").sum()),
         "copyright": int((d.cls == "copyright").sum()),
         "private": int((d.cls == "private").sum()),
         "exists": int(ex.sum())}
    n = int(len(d))
    assert sum(c.values()) == n, (sum(c.values()), n)
    return {"n": n, "count": c, "share_pct": {k: round(100 * v / n, 1) if n else None for k, v in c.items()},
            "not_exists_pct": round(100 * (n - c["exists"]) / n, 1) if n else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, default=ROOT / "data/probe/missing_reasons_20261003_230351.jsonl")
    ap.add_argument("--ledger", type=Path, default=ROOT / "state/panel_cohort.json")
    ap.add_argument("--raw-glob", default=str(ROOT / "data/raw/daily_2026*.jsonl"))
    ap.add_argument("--seed-age-max", type=float, default=1.0, help="first observation within this many days of publication")
    ap.add_argument("--done-days", type=float, default=35.0)
    args = ap.parse_args()

    df = pd.read_json(args.probe, lines=True).set_index("video_id")
    led = pd.DataFrame.from_dict(json.load(open(args.ledger, encoding="utf-8"))["videos"], orient="index")
    led["published_at"] = pd.to_datetime(led.published_at, utc=True)
    led["last"] = pd.to_datetime(led.last_observed_at, utc=True, format="ISO8601")
    end = led["last"].max()
    meta = m28.load_meta(args.raw_glob).set_index("video_id") if "video_id" in m28.load_meta(args.raw_glob).columns else m28.load_meta(args.raw_glob)
    miss = df[df.group == "missing"].join(led[["published_at", "seed_age_days"]], rsuffix="_led").join(meta[["ch_country"]], how="left")
    miss["seg"] = m28.segment(miss.ch_country)
    early = miss[miss.seed_age_days <= args.seed_age_max]
    done = early[(end - early.published_at).dt.total_seconds() / 86400 >= args.done_days]
    ctrl = df[df.group == "control"]
    out = {"probe_file": str(args.probe), "panel_end": end.isoformat(), "rows": ROWS,
           "definitions": {"early": f"missing & seed_age_days<={args.seed_age_max}", "done": f"early & elapsed>={args.done_days} d at panel_end",
                           "exists": "has_video_details==True", "no_reason": "cls==unavailable_generic & has_video_details==False"},
           "all_missing": table(miss), "early": table(early), "early_done": table(done),
           "early_by_segment": {s: table(early[early.seg == s]) for s in m28.SEG_ORDER},
           "early_by_category": {c: table(g) for c, g in early.groupby("category_name")},
           "control": {"n": int(len(ctrl)), "exists": int(ctrl.has_video_details.astype(bool).sum())}}
    op = Path(str(args.probe).replace(".jsonl", "_table9.json"))
    op.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"early": out["early"]["share_pct"] | {"n": out["early"]["n"]}, "early_done": out["early_done"]["share_pct"] | {"n": out["early_done"]["n"]},
                      "control": out["control"]}, ensure_ascii=False))
    print("wrote", op)
    return 0


if __name__ == "__main__":
    sys.exit(main())
