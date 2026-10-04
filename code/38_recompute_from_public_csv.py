#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""38_recompute_from_public_csv.py -- Recompute the article's main point estimates from the public derived data alone.

Needs only the three CSV files of the public package (no raw daily observations, no API access):
  data/per_video_metrics.csv      26,705-video analytic sample (S1 Data)
  data/disappearance_cohort.csv   24-hour cohort with disappearance status and event times
  data/video_ids.csv              all tracked videos (used only for counts)

Reproduces: per-video median and view-weighted cumulative shares (Table 3, S8k), cumulative and increment regressions
(Table 4), late growth (Table 5), like-rate ratios and the stage decomposition (Table 7, S8j), the like-count-not-retrieved
share, disappearance rates, Kaplan-Meier estimates and the displayed-reason table (Tables 8, 9, S9), the survivorship counts
(S8f) and the shares by creator location and category (Tables 3, 6, 8). Channel-cluster bootstrap intervals for the headline
estimates are optional (--boot N) and use channel_key.

What cannot be recomputed from the derived data: anything that needs the daily observations themselves, that is the
interpolation and monotonisation sensitivity (S5, S8b), the day-3 views of videos lost before day 28 (S8f, two rows), the
Music title-word split, the within-channel variance decomposition, and the observation-schedule sensitivity (S8i).

Usage: python code/38_recompute_from_public_csv.py [--repo <package folder>] [--boot 0] [--out results/recomputed_from_public_csv.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PTS = [7, 14, 21, 28, 35]
SEGS = ["JP", "US", "IN", "other", "unknown"]


def km(times, events, grid):
    times = np.asarray(times, float); events = np.asarray(events, int)
    ev_t = np.unique(times[events == 1]); S = 1.0; keys, vals = [], []
    for ut in ev_t:
        at_risk = int(np.sum(times >= ut)); d = int(np.sum((times == ut) & (events == 1)))
        if at_risk > 0:
            S *= (1 - d / at_risk)
        keys.append(ut); vals.append(S)
    keys = np.array(keys); vals = np.array(vals)
    return [0.0 if (k := np.searchsorted(keys, g, side="right")) == 0 else 1 - float(vals[k - 1]) for g in grid]


def r2_slope(x, y):
    b, a = np.polyfit(x, y, 1); yh = a + b * x
    return float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)), float(b)


def cboot(values, clusters, stat, B, rng):
    values = np.asarray(values, float); clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    idx = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        out.append(stat(values[np.concatenate([idx[p] for p in pick])]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent, help="package folder containing data/ (default: parent of code/)")
    ap.add_argument("--boot", type=int, default=0, help="channel-cluster bootstrap replicates for the headline estimates (0 = none)")
    ap.add_argument("--out", type=Path, default=None, help="JSON output (default: <repo>/results/recomputed_from_public_csv.json)")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    m = pd.read_csv(args.repo / "data/per_video_metrics.csv", low_memory=False)
    d = pd.read_csv(args.repo / "data/disappearance_cohort.csv", low_memory=False)
    ids = pd.read_csv(args.repo / "data/video_ids.csv", low_memory=False)
    R = {"n_tracked_videos": int(len(ids)), "n_analytic": int(len(m)), "n_channels_analytic": int(m.channel_key.nunique())}

    # ---- view dynamics (Tables 3, 4, 5; S8k)
    R["share_median_pct"] = {f"day{t}": round(100 * m[f"sV{t}"].median(), 1) for t in [1, 3, 7, 14]}
    R["share_view_weighted_pct"] = {f"day{t}": round(100 * m[f"V{t}"].sum() / m.V28.sum(), 1) for t in [1, 2, 3, 7, 14]}
    R["median_V28"] = float(m.V28.median())
    R["top1pct_share_of_views"] = round(float(m.V28.sort_values(ascending=False).head(round(0.01 * len(m))).sum() / m.V28.sum()), 3)
    reg = {}
    for t in [1, 2, 3, 7, 14]:
        x = np.log10(m[f"V{t}"].clip(lower=1)); y = np.log10(m.V28.clip(lower=1))
        r2, b = r2_slope(x.to_numpy(), y.to_numpy()); reg[f"day{t}"] = {"R2_cumulative": round(r2, 3), "slope_cumulative": round(b, 3)}
        gain = (m.V28 - m[f"V{t}"]).clip(lower=0); pos = gain > 0
        if t <= 7:
            r2i, bi = r2_slope(x[pos].to_numpy(), np.log10(gain[pos] + 1).to_numpy())
            reg[f"day{t}"].update({"n_positive_increment": int(pos.sum()), "R2_increment": round(r2i, 3), "slope_increment": round(bi, 3)})
    R["regressions"] = reg
    R["late_growth_pct"] = {"ge2x": round(100 * m.late_growth_ge2x.mean(), 1), "ge1_5x": round(100 * m.late_growth_ge1_5x.mean(), 1), "ge3x": round(100 * m.late_growth_ge3x.mean(), 1),
                            "reaccelerated": round(100 * m.reaccelerated.mean(), 1), "median_ratio_V28_over_V7": round(float(m.ratio_V28_over_V7.median()), 3)}
    R["late_ge2x_by_day7_quintile_pct"] = {str(k): round(100 * v, 1) for k, v in m.groupby("day7_view_quintile").late_growth_ge2x.mean().items()}
    R["share_day3_by_category_pct"] = {k: round(100 * v, 1) for k, v in m.groupby("category").sV3.median().sort_values().items()}
    R["share_day3_by_subscriber_quintile_pct"] = {str(k): round(100 * v, 1) for k, v in m.groupby("subscriber_quintile").sV3.median().items()}
    R["share_day3_by_location_pct"] = {k: round(100 * m.loc[m.creator_location == k, "sV3"].median(), 1) for k in SEGS}
    R["median_V28_by_location"] = {k: float(m.loc[m.creator_location == k, "V28"].median()) for k in SEGS}

    # ---- like rate (Table 7; S8j) and like-count-not-retrieved share
    both = m.dropna(subset=["like_rate_early", "like_rate_late"])
    R["like_ratio"] = {"n": int(len(both)), "median_late_over_early": round(float(both.like_ratio_late_over_early.median()), 3),
                       "share_declined_pct": round(100 * float((both.like_ratio_late_over_early < 1).mean()), 1),
                       "median_early_pct": round(100 * float(both.like_rate_early.median()), 2), "median_late_pct": round(100 * float(both.like_rate_late.median()), 2),
                       "by_day28_quintile": {str(k): round(float(v), 3) for k, v in both.groupby("day28_view_quintile").like_ratio_late_over_early.median().items()},
                       "by_location": {k: round(float(both.loc[both.creator_location == k, "like_ratio_late_over_early"].median()), 3) for k in SEGS}}
    stages = ["like_rate_early", "like_rate_day1", "like_rate_day3", "like_rate_day7", "like_rate_day28", "like_rate_late"]
    g = m.dropna(subset=stages); g = g[g.like_rate_early_age_days < 0.5]
    pairs = [("like_rate_early", "like_rate_day1"), ("like_rate_day1", "like_rate_day3"), ("like_rate_day3", "like_rate_day7"), ("like_rate_day7", "like_rate_day28"),
             ("like_rate_day28", "like_rate_late"), ("like_rate_day1", "like_rate_day28"), ("like_rate_early", "like_rate_late")]
    R["like_rate_stages"] = {"n": int(len(g)), "median_age_first_observation_days": round(float(g.like_rate_early_age_days.median()), 2),
                             "median_ratios": {f"{b.replace('like_rate_', '')}/{a.replace('like_rate_', '')}": round(float((g[b] / g[a]).median()), 3) for a, b in pairs}}
    R["like_count_not_retrieved_pct"] = {"all": round(100 * m.likes_hidden.mean(), 1), **{k: round(100 * m.loc[m.creator_location == k, "likes_hidden"].mean(), 1) for k in ["JP", "US", "IN"]}}

    # ---- disappearance (Tables 8, 9, S9; survivorship counts S8f)
    c = d[d.in_panel_24h_cohort == True].copy()
    c["disappeared_v2"] = c.disappeared_v2.astype(bool)
    c35 = c[c.elapsed_days_at_panel_end >= 35]; c7 = c[c.elapsed_days_at_panel_end >= 7]
    ev7 = (c7.disappeared_v2 & (c7.first_failed_retrieval_age_days <= 7))
    R["disappearance"] = {"n_24h_cohort": int(len(c)), "n_passed_35d": int(len(c35)), "events_35d": int(c35.disappeared_v2.sum()),
                          "events_three_misses": int(c35.disappeared_three_misses.fillna(False).astype(bool).sum()), "events_probe_confirmed": int(c35.disappeared_confirmed_by_probe.fillna(False).astype(bool).sum()),
                          "rate_35d_pct": round(100 * c35.disappeared_v2.mean(), 1), "rate_35d_three_misses_only_pct": round(100 * c35.disappeared_three_misses.fillna(False).astype(bool).mean(), 1),
                          "n_passed_7d": int(len(c7)), "rate_7d_pct": round(100 * ev7.mean(), 1),
                          "km_pct": dict(zip([f"day{p}" for p in PTS], [round(100 * v, 1) for v in km(c.km_time_days, c.disappeared_v2.astype(int), PTS)])),
                          "km_at_risk": {f"day{p}": int((c.km_time_days >= p).sum()) for p in [0] + PTS},
                          "by_location_pct": {k: round(100 * c35.loc[c35.creator_location == k, "disappeared_v2"].mean(), 1) for k in SEGS},
                          "n_by_location": {k: int((c35.creator_location == k).sum()) for k in SEGS},
                          "by_category_pct": {k: round(100 * v, 1) for k, v in c35.groupby("category_name").disappeared_v2.mean().items() if (c35.category_name == k).sum() >= 300},
                          "km_by_location_pct": {k: dict(zip([f"day{p}" for p in PTS], [round(100 * v, 1) for v in km(c.loc[c.creator_location == k, "km_time_days"], c.loc[c.creator_location == k, "disappeared_v2"].astype(int), PTS)])) for k in SEGS},
                          "median_first_failed_retrieval_age_days": round(float(c35.loc[c35.disappeared_v2, "first_failed_retrieval_age_days"].median()), 1)}
    led35 = d[(d.in_ledger_24h_cohort == True) & (d.in_35d_cohort == True)]
    R["disappearance"]["three_miss_rule_ledger_cohort"] = {"n_passed_35d": int(len(led35)), "events": int(led35.disappeared.astype(bool).sum()), "rate_pct": round(100 * led35.disappeared.astype(bool).mean(), 1)}
    ev_all = c[c.disappeared_v2]
    t9 = ev_all.probe_class.value_counts(dropna=False); t9_35 = c35[c35.disappeared_v2].probe_class.value_counts(dropna=False)
    R["displayed_reasons"] = {"n": int(len(ev_all)), "counts": {str(k): int(v) for k, v in t9.items()}, "shares_pct": {str(k): round(100 * v / len(ev_all), 1) for k, v in t9.items()},
                              "n_35d": int(t9_35.sum()), "shares_35d_pct": {str(k): round(100 * v / t9_35.sum(), 1) for k, v in t9_35.items()}}
    e28 = c[c.elapsed_days_at_panel_end >= 28]
    lost = e28[e28.disappeared_v2 & (e28.first_failed_retrieval_age_days < 28)]
    R["survivorship"] = {"n_passed_28d": int(len(e28)), "lost_before_28d": int(len(lost)), "share_pct": round(100 * len(lost) / len(e28), 1),
                         "lost_first_failed_before_3d": int((lost.first_failed_retrieval_age_days < 3).sum()),
                         "lost_music_pct": round(100 * float((lost.category_name == "Music").mean()), 1), "lost_india_pct": round(100 * float((lost.creator_location == "IN").mean()), 1),
                         "analytic_music_pct": round(100 * float((m.category == "Music").mean()), 1), "analytic_india_pct": round(100 * float((m.creator_location == "IN").mean()), 1)}

    # ---- optional bootstrap for the headline estimates
    if args.boot > 0:
        R["bootstrap"] = {"replicates": args.boot,
                          "share_day3_median_pct_95ci": [round(100 * v, 1) for v in cboot(m.sV3, m.channel_key, np.median, args.boot, rng)],
                          "rate_35d_pct_95ci": [round(100 * v, 1) for v in cboot(c35.disappeared_v2.astype(float), c35.channel_key, np.mean, args.boot, rng)],
                          "like_ratio_median_95ci": [round(v, 3) for v in cboot(both.like_ratio_late_over_early, both.channel_key, np.median, args.boot, rng)]}

    out = args.out or (args.repo / "results" / "recomputed_from_public_csv.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: R[k] for k in ["n_analytic", "share_median_pct", "share_view_weighted_pct", "late_growth_pct", "like_ratio", "like_rate_stages", "disappearance", "survivorship"]}, ensure_ascii=False, indent=1, default=str)[:6000])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
