#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""37_review2_reanalysis.py -- Re-analyses requested by the second pre-submission review (2026-10-04, after revision).

A. Survivorship with the revised event definition (event time = first failed retrieval; window-end events confirmed on the
   watch page included), counts, composition and the day-3 comparison on the same set of videos.
B. View-weighted (sum) cumulative shares, as a contrast to the per-video median.
C. Like-rate change decomposed within one group of videos (first observation -> day 1 -> day 3 -> day 7 -> day 28 -> late).
D. Timing of disappearance events: gap between last success and first failure, the window-end probe-confirmed events,
   observation-run gaps, Kaplan-Meier restricted to videos published after daily operation became stable (2026-07-14),
   and interval bounds for the event time (last success / midpoint / first failure).
E. Joint standardisation on common support (strata with >= 5 videos in every compared location).
F. Kaplan-Meier confidence bands over the full grid and numbers at risk, for Fig 6 and S9 Table.

Usage: python scripts/37_review2_reanalysis.py [--boot 1000] [--km-boot 200]
Input: data/panel/panel_observations.jsonl, reports/panel_lifecycle_*_traj.parquet, the latest reports/review_reanalysis_*.json
       and the per-video parquet files it names.
Output: reports/review2_reanalysis_<ts>.json
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


m28 = load_mod("m28", HERE / "28_panel_lifecycle.py")
m36 = load_mod("m36", HERE / "36_review_reanalysis.py")
cboot, km, km_boot, rate_table = m36.cboot, m36.km, m36.km_boot, m36.rate_table
STEM = "panel_lifecycle_20261003_231558"
SEG_ORDER = m28.SEG_ORDER
PTS = [7, 14, 21, 28, 35]
GRID = list(range(0, 36))


def log(*a):
    print(datetime.now().strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def ci3(t):
    return {"est": float(t[0]), "lo": float(t[1]), "hi": float(t[2])}


def km_band(d, B, rng):
    """KM estimate with channel-bootstrap percentile band over GRID, plus the values at PTS."""
    b = km_boot(d, GRID, GRID, B, rng)
    est = [b[str(g)]["est"] for g in GRID]; lo = [b[str(g)]["lo"] for g in GRID]; hi = [b[str(g)]["hi"] for g in GRID]
    return {"est": est, "lo": lo, "hi": hi, "at_points": {str(p): b[str(p)] for p in PTS},
            "at_risk": {str(p): int((d.time >= p).sum()) for p in [0] + PTS}, "n": int(len(d)), "events": int(d.event.sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--km-boot", type=int, default=200)
    args = ap.parse_args()
    rng = np.random.default_rng(0); B = args.boot; KB = args.km_boot
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT = {"generated_at": ts, "boot": B, "km_boot": KB}

    log("loading ...")
    df = m28.load_panel(ROOT / "data/panel/panel_observations.jsonl")
    df["observed_at"] = pd.to_datetime(df.observed_at, utc=True); df["published_at"] = pd.to_datetime(df.published_at, utc=True)
    traj = pd.read_parquet(ROOT / "reports" / f"{STEM}_traj.parquet"); traj.index.name = "video_id"
    rv = sorted(glob.glob(str(ROOT / "reports/review_reanalysis_*.json")))[-1]
    V = json.load(open(rv, encoding="utf-8")); OUT["review_json"] = Path(rv).name
    E = pd.read_parquet(ROOT / V["disappearance_v2"]["per_video_file"])
    pv = pd.read_parquet(ROOT / V["like_rate_v2"]["per_video_file"])
    pub = df.groupby("video_id").published_at.first()
    E = E.join(pub.rename("pub"), how="left")
    E["seed_month"] = E.pub.dt.tz_convert("Asia/Tokyo").dt.strftime("%Y-%m")
    E["event"] = E.event.astype(int)
    run_times = np.sort(df.groupby("run_id").observed_at.first().to_numpy())

    # ---------------------------------------------------------------- A. survivorship, revised event definition
    log("A. survivorship ...")
    elig = E[E.elapsed_end >= 28]
    lost = elig[(elig.event == 1) & (elig.first_miss_age < 28)]
    lost_3m = elig[(elig.event_confirmed) & (elig.first_miss_age < 28)]
    lost_old = elig[(elig.event_confirmed) & (elig.last_age < 28)]
    sub = df[df.video_id.isin(lost.index)].sort_values(["video_id", "age_days"])
    v3 = {}
    for vid, g in sub.groupby("video_id", sort=False):
        a = g.age_days.to_numpy(float); v = np.maximum.accumulate(np.maximum(g.views.to_numpy(float), 0))
        if len(a) >= 2 and a.min() <= 3 <= a.max():
            v3[vid] = float(np.expm1(np.interp(3, a, np.log1p(v))))
    v3 = pd.Series(v3, dtype=float)
    analytic_ids = elig.index.intersection(traj.index)
    comp = {}
    for col in ["category", "seg", "seed_month"]:
        comp[col] = {"lost": lost[col].value_counts(normalize=True).round(4).to_dict(),
                     "analytic": E.loc[analytic_ids, col].value_counts(normalize=True).round(4).to_dict()}
    OUT["survivorship_v3"] = {
        "definition": "videos first observed <=24 h, >=28 d elapsed at panel end; lost = disappearance event (three misses or probe-confirmed) with first failed retrieval before day 28",
        "eligible_28d": int(len(elig)), "lost_before_28d": int(len(lost)), "share_lost": float(len(lost) / len(elig)),
        "lost_before_28d_three_misses_only": int(len(lost_3m)), "lost_before_28d_old_definition_last_success_lt_28": int(len(lost_old)),
        "lost_first_miss_before_3d": int((lost.first_miss_age < 3).sum()), "lost_last_success_before_3d": int((lost.last_age < 3).sum()),
        "lost_with_V3": int(len(v3)), "medV3_lost": float(v3.median()) if len(v3) else None, "medV3_analytic": float(traj.V3.median()),
        "V3_lost_q": v3.quantile([.25, .5, .75]).round(0).to_dict() if len(v3) else None, "V3_analytic_q": traj.V3.quantile([.25, .5, .75]).round(0).to_dict(),
        "lost_first_miss_age_q": lost.first_miss_age.quantile([.1, .25, .5, .75, .9]).round(2).to_dict(),
        "first_age_median": {"lost": float(lost.first_age.median()), "analytic": float(E.loc[analytic_ids].first_age.median())},
        "composition": comp}
    # the same set of videos, V3 comparison restricted to those lost after day 3 with V3
    OUT["survivorship_v3"]["medV3_lost_ci"] = ci3(cboot(v3.to_numpy(), lost.loc[v3.index].channel_id.to_numpy(), np.nanmedian, min(B, 500), rng)) if len(v3) else None

    # ---------------------------------------------------------------- B. view-weighted shares
    log("B. sum-weighted shares ...")
    Bs = {f"sum_V{t}_over_sum_V28": float(traj[f"V{t}"].sum() / traj.V28.sum()) for t in [1, 2, 3, 7, 14]}
    Bs["median_sV3"] = float(traj.sV3.median()); Bs["median_sV7"] = float(traj.sV7.median())
    Bs["median_sV3_by_V28q"] = {str(k): float(v) for k, v in traj.groupby("V28q", observed=True).sV3.median().items()}
    Bs["share_of_views_top1pct"] = float(traj.V28.sort_values(ascending=False).head(int(round(len(traj) * 0.01))).sum() / traj.V28.sum())
    OUT["view_weighted_shares"] = Bs

    # ---------------------------------------------------------------- C. like-rate change within one group
    log("C. like-rate decomposition ...")
    L = pv.join(traj[["V28q", "channel"]], how="left")
    stages = ["lr_early", "lr_day1", "lr_day3", "lr_day7", "lr_day28", "lr_late"]
    pairs = [("lr_early", "lr_day1"), ("lr_day1", "lr_day3"), ("lr_day3", "lr_day7"), ("lr_day7", "lr_day28"), ("lr_day28", "lr_late"),
             ("lr_early", "lr_day28"), ("lr_day1", "lr_day28"), ("lr_early", "lr_late")]
    G2 = L.dropna(subset=stages)
    G = G2[G2.age_early < 0.5]  # the first observation is strictly before the day-1 window, so each stage is a different observation
    def ratios(d, with_ci):
        o = {}
        for a, b in pairs:
            r = (d[b] / d[a]).to_numpy()
            o[f"{b}/{a}"] = ci3(cboot(r, d.channel.to_numpy(), np.nanmedian, 300, rng)) if with_ci else {"est": float(np.nanmedian(r))}
        return o
    C = {"stages": stages, "n": int(len(G)), "age_early_median": float(G.age_early.median()), "age_late_median": float(G.age_late.median()),
         "medians": {s: float(G[s].median()) for s in stages}, "ratios": ratios(G, True),
         "share_declined_early_to_day1": float((G.lr_day1 < G.lr_early).mean()), "share_declined_day1_to_day28": float((G.lr_day28 < G.lr_day1).mean()),
         "by_V28q": {str(q): {"n": int(len(d)), "medians": {s: float(d[s].median()) for s in stages}, "ratios": ratios(d, False)} for q, d in G.groupby("V28q", observed=True)},
         "n_without_age_restriction": int(len(G2)), "ratios_without_age_restriction": ratios(G2, False)}
    OUT["like_rate_decomposition"] = C

    # ---------------------------------------------------------------- D. timing of disappearance events
    log("D. event timing ...")
    ev35 = E[(E.elapsed_end >= 35) & (E.event == 1)]
    gap = ev35.first_miss_age - ev35.last_age
    D = {"n_events_35d": int(len(ev35)), "gap_q": gap.quantile([.5, .9, .95, .99]).round(3).to_dict(), "gap_max": float(gap.max()),
         "gap_gt_1_5d": int((gap > 1.5).sum()), "gap_gt_7d": int((gap > 7).sum()), "gap_gt_20d": int((gap > 20).sum()),
         "gap_gt_1_5d_by_seed_month": ev35[gap > 1.5].seed_month.value_counts().to_dict(),
         "gap_gt_1_5d_last_success_dates": sorted(set((ev35[gap > 1.5].pub + pd.to_timedelta(ev35[gap > 1.5].last_age, unit="D")).dt.tz_convert("Asia/Tokyo").dt.strftime("%Y-%m-%d")))}
    pr = ev35[ev35.event_probe]
    D["probe_confirmed_35d"] = {"n": int(len(pr)), "first_miss_age_min": float(pr.first_miss_age.min()), "first_miss_age_max": float(pr.first_miss_age.max()),
                                "first_miss_age_q": pr.first_miss_age.quantile([.1, .5, .9]).round(2).to_dict(), "by_seed_month": pr.seed_month.value_counts().to_dict(),
                                "n_first_miss_before_33d": int((pr.first_miss_age < 33).sum()), "n_gap_gt_1_5d": int(((pr.first_miss_age - pr.last_age) > 1.5).sum()),
                                "n_last_success_ge_33d": int((pr.last_age >= 33).sum())}
    rt = pd.Series(run_times); rg = rt.diff().dt.total_seconds() / 86400
    D["n_runs"] = int(len(rt)); D["run_gaps_gt_1_5d"] = [{"from": str(rt[i - 1].date()), "to": str(rt[i].date()), "days": round(float(rg[i]), 2)} for i in range(1, len(rt)) if rg[i] > 1.5]
    D["last_run_gap_gt_1_5d_ends"] = D["run_gaps_gt_1_5d"][-1]["to"] if D["run_gaps_gt_1_5d"] else None
    # stable period: published on or after 2026-07-14 JST (daily operation from 2026-07-14)
    cut = pd.Timestamp("2026-07-14", tz="Asia/Tokyo")
    Es = E[E.pub >= cut].copy(); Eb = E[E.pub < cut].copy()
    def rates(d):
        d35 = d[d.elapsed_end >= 35].copy(); d35["miss"] = d35.event.astype(float)
        d7 = d[d.elapsed_end >= 7].copy(); d7["miss7"] = ((d7.event == 1) & (d7.first_miss_age <= 7)).astype(float)
        return {"n": int(len(d)), "events": int(d.event.sum()), "n_35d": int(len(d35)), "rate_35d": rate_table(d35, None, "miss", "channel_id", B, rng)[0],
                "n_7d": int(len(d7)), "rate_7d": rate_table(d7, None, "miss7", "channel_id", B, rng)[0], "km": km_band(d, KB, rng)}
    D["stable_period"] = {"definition": "published on or after 2026-07-14 00:00 JST", **rates(Es)}
    D["before_stable_period"] = {"definition": "published before 2026-07-14 00:00 JST", **rates(Eb)}
    # interval bounds for the event time
    t_first = E.time.to_numpy(float)
    t_last = np.where(E.event == 1, E.last_age.to_numpy(float), t_first)
    t_mid = np.where(E.event == 1, (E.last_age.to_numpy(float) + E.first_miss_age.to_numpy(float)) / 2, t_first)
    D["event_time_bounds_km"] = {"first_failed_retrieval (main)": dict(zip(map(str, PTS), km(t_first, E.event, PTS))),
                                 "last_successful_retrieval (lower bound)": dict(zip(map(str, PTS), km(t_last, E.event, PTS))),
                                 "midpoint": dict(zip(map(str, PTS), km(t_mid, E.event, PTS)))}
    OUT["event_timing"] = D

    # ---------------------------------------------------------------- E. joint standardisation on common support
    log("E. common support ...")
    cell = traj.groupby(["category", "subq"], observed=True).size(); w = cell / cell.sum()
    sizes = traj.groupby(["seg", "category", "subq"], observed=True).size().unstack("seg").fillna(0)
    sizes = sizes.reindex(cell.index).fillna(0)
    common3 = sizes.index[(sizes[["JP", "US", "IN"]] >= 5).all(axis=1)]
    common5 = sizes.index[(sizes[SEG_ORDER] >= 5).all(axis=1)]
    def std_on(d, strata):
        med = d.groupby(["category", "subq"], observed=True).sV3.median().reindex(strata)
        ww = w.reindex(strata).fillna(0); ok = med.notna()
        return float((ww[ok] * med[ok]).sum() / ww[ok].sum())
    Ecs = {"n_strata_total": int(len(cell))}
    for label, strata in [("common_JP_US_IN", common3), ("common_all_five", common5)]:
        res = {"n_strata": int(len(strata)), "weight_coverage": float(w.reindex(strata).sum())}
        for s in SEG_ORDER:
            d = traj[traj.seg == s]; est = std_on(d, strata)
            uniq = d.channel.unique(); groups = {c: idx for c, idx in d.groupby("channel").groups.items()}
            reps = []
            for _ in range(300):
                pick = rng.choice(uniq, len(uniq), replace=True)
                reps.append(std_on(d.loc[np.concatenate([groups[c] for c in pick])], strata))
            res[s] = {"est": est, "lo": float(np.percentile(reps, 2.5)), "hi": float(np.percentile(reps, 97.5)), "n": int(len(d))}
        Ecs[label] = res
    jp = sizes["JP"]; Ecs["JP_strata_below_5"] = {"n": int((jp < 5).sum()), "weight": float(w[jp < 5].sum()), "strata": [f"{c} x {q}" for c, q in jp[jp < 5].index]}
    Ecs["original_all_strata"] = {s: V["joint_standardisation_sV3"][s] for s in SEG_ORDER}
    OUT["joint_standardisation_common_support"] = Ecs

    # ---------------------------------------------------------------- F. KM bands and numbers at risk
    log("F. KM bands ...")
    F = {"grid": GRID, "points": PTS, "all": km_band(E, KB, rng)}
    for s in SEG_ORDER:
        F[s] = km_band(E[E.seg == s], KB, rng)
    OUT["km_bands"] = F

    outp = ROOT / "reports" / f"review2_reanalysis_{ts}.json"
    outp.write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log("wrote", outp)
    summary = {"out": str(outp),
               "survivorship": {k: OUT["survivorship_v3"][k] for k in ["eligible_28d", "lost_before_28d", "lost_before_28d_old_definition_last_success_lt_28", "lost_first_miss_before_3d", "lost_with_V3", "medV3_lost", "medV3_analytic"]},
               "sum_shares": {k: round(v, 4) for k, v in Bs.items() if k.startswith("sum_")},
               "like_decomp": {"n": C["n"], **{k: round(v["est"], 3) for k, v in C["ratios"].items()}},
               "gap": {"gt_1_5d": D["gap_gt_1_5d"], "gt_20d": D["gap_gt_20d"], "probe": D["probe_confirmed_35d"]["by_seed_month"], "first_miss_min": D["probe_confirmed_35d"]["first_miss_age_min"]},
               "stable_km35": D["stable_period"]["km"]["at_points"]["35"], "stable_rate35": D["stable_period"]["rate_35d"]["est"], "bounds": D["event_time_bounds_km"],
               "common3": {s: round(Ecs["common_JP_US_IN"][s]["est"], 4) for s in SEG_ORDER}, "common3_cov": Ecs["common_JP_US_IN"]["weight_coverage"],
               "run_gaps": D["run_gaps_gt_1_5d"]}
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
