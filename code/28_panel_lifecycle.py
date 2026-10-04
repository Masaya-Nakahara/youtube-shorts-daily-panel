#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""28_panel_lifecycle.py -- Lifecycle analysis of the daily Shorts panel.

Reads the irreplaceable panel (data/panel/panel_observations.jsonl), the cohort
ledger (state/panel_cohort.json) and the raw daily search files (creator country,
seed search order, title), builds the analytic cohort of videos first observed
within 24 h of publication and followed to >= 28 days, and computes:

  * lifecycle shape  : share of day-28 views reached by day 1/3/7/14, power-law slope
  * predictability   : log V28 ~ log V_t fits, top-decile persistence
  * late growth      : share with V28/V7 >= 2, re-acceleration after day 7
  * heterogeneity    : by category, creator-country segment, seed order, duration
  * engagement       : within-video like-rate dilution (early vs late)
  * disappearance    : 7-day / 35-day disappearance among seeded videos
  * within-channel variance decomposition

All group estimates carry channel-cluster bootstrap 95% CIs (resampling channels
with replacement). Outputs: <stem>.json (all numbers), <stem>_tables.md (paper
tables), figures in --figdir (PNG, English labels). Pure computation, no API quota.

Usage:
  python scripts/28_panel_lifecycle.py [--boot 1000] [--seed 0]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.json as pj
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
AGES = [1, 2, 3, 5, 7, 10, 14, 21, 28]
CAT_MIN_N = 300
SEG_ORDER = ["JP", "US", "IN", "other", "unknown"]
SEG_LABEL = {"JP": "Japan-based", "US": "US-based", "IN": "India-based", "other": "Other known", "unknown": "Unknown"}
# validated categorical palette (dataviz skill, light mode)
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- loading
def load_panel(path: Path) -> pd.DataFrame:
    tbl = pj.read_json(str(path), read_options=pj.ReadOptions(block_size=64 << 20))
    df = tbl.to_pandas()
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    return df


def load_meta(raw_glob: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(raw_glob)):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sn = d.get("snippet", {}) or {}
                ch = d.get("_channel") or {}
                chs = ch.get("statistics", {}) or {}
                ctx = d.get("_search_context", {}) or {}
                rows.append((d.get("id"), (ch.get("snippet", {}) or {}).get("country") or "NA",
                             sn.get("defaultAudioLanguage") or sn.get("defaultLanguage") or "",
                             ctx.get("order"), int(chs.get("subscriberCount") or 0), sn.get("title", "")))
    meta = pd.DataFrame(rows, columns=["video_id", "ch_country", "lang", "order", "subs", "title"])
    return meta.drop_duplicates("video_id").set_index("video_id")


def segment(cc: pd.Series) -> pd.Series:
    cc = cc.fillna("NA")
    return pd.Series(np.select([cc == "JP", cc == "US", cc == "IN", cc == "NA"], ["JP", "US", "IN", "unknown"], "other"), index=cc.index)


# ----------------------------------------------------------------------------- bootstrap
def cluster_boot(values: np.ndarray, clusters: np.ndarray, stat, B: int, rng, weights=None) -> tuple[float, float, float]:
    """Channel-cluster bootstrap: resample clusters with replacement, recompute stat(values_sub)."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    idx_by_c = [[] for _ in uniq]
    for i, c in enumerate(inv):
        idx_by_c[c].append(i)
    idx_by_c = [np.asarray(x) for x in idx_by_c]
    point = stat(values)
    if len(uniq) < 2:
        return point, np.nan, np.nan
    out = np.empty(B)
    n = len(uniq)
    for b in range(B):
        draw = rng.integers(0, n, n)
        sel = np.concatenate([idx_by_c[c] for c in draw])
        out[b] = stat(values[sel])
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return point, lo, hi


def boot_table(df: pd.DataFrame, by: str | None, col: str, stat, B: int, rng, order=None, min_n=1) -> pd.DataFrame:
    rows = []
    groups = [(None, df)] if by is None else list(df.groupby(by, observed=True))
    for key, g in groups:
        if len(g) < min_n:
            continue
        v = g[col].to_numpy(dtype=float)
        pt, lo, hi = cluster_boot(v, g["channel"].to_numpy(), stat, B, rng)
        rows.append({"group": "all" if key is None else key, "n": len(g), "n_channels": g["channel"].nunique(), "est": pt, "lo": lo, "hi": hi})
    t = pd.DataFrame(rows)
    if order is not None and by is not None:
        t["__o"] = t["group"].map({k: i for i, k in enumerate(order)})
        t = t.sort_values("__o").drop(columns="__o")
    return t


def nanmedian(x):
    return float(np.nanmedian(x))


def mean_(x):
    return float(np.nanmean(x))


def fmt_pct(e, lo, hi, d=1):
    return f"{100*e:.{d}f} [{100*lo:.{d}f}, {100*hi:.{d}f}]"


def fmt_num(e, lo, hi, d=2):
    return f"{e:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, default=ROOT / "data/panel/panel_observations.jsonl")
    ap.add_argument("--ledger", type=Path, default=ROOT / "state/panel_cohort.json")
    ap.add_argument("--raw-glob", default=str(ROOT / "data/raw/daily_2026*.jsonl"))
    ap.add_argument("--out-stem", type=Path, default=None)
    ap.add_argument("--figdir", type=Path, default=ROOT / "reports/figures/panel")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = args.out_stem or (ROOT / "reports" / f"panel_lifecycle_{ts}")
    args.figdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    B = args.boot
    R: dict = {"generated_at": ts, "boot": B, "seed": args.seed}

    log("loading panel ...")
    df = load_panel(args.panel)
    meta = load_meta(args.raw_glob)
    led = pd.DataFrame.from_dict(json.load(open(args.ledger, encoding="utf-8"))["videos"], orient="index")
    log(f"panel rows={len(df)} videos={df.video_id.nunique()} channels={df.channel_id.nunique()}")

    # ---- panel description
    runs = df.groupby("run_id").agg(n=("video_id", "size"), date=("observed_at", "first"))
    runs["date"] = runs["date"].dt.date
    run_dates = sorted(set(runs.date))
    all_days = pd.date_range(run_dates[0], run_dates[-1]).date
    R["panel"] = {
        "rows": int(len(df)), "videos": int(df.video_id.nunique()), "channels": int(df.channel_id.nunique()),
        "runs": int(len(runs)), "first_run": str(run_dates[0]), "last_run": str(run_dates[-1]),
        "run_days": len(run_dates), "missing_days": [str(d) for d in all_days if d not in set(run_dates)],
        "regions": df.region.value_counts().to_dict(),
        "ledger_total": int(len(led)), "ledger_retired_reason": led.retired_reason.fillna("active").value_counts().to_dict(),
        "seed_age_q": led.seed_age_days.quantile([.1, .5, .9]).round(3).to_dict(),
    }

    # ---- per-video summary
    g = df.groupby("video_id")
    per = g.agg(n_obs=("views", "size"), first_age=("age_days", "min"), last_age=("age_days", "max"),
                category=("category_name", "first"), channel=("channel_id", "first"), dur=("duration_s", "first"),
                pub=("published_at", "first"))
    per = per.join(meta[["ch_country", "lang", "order", "subs"]], how="left")
    per["seg"] = segment(per.ch_country)
    all_first1 = per[per.first_age <= 1.0]
    R["cohort_funnel"] = {
        "tracked_videos": int(len(per)),
        "first_obs_within_24h": int(len(all_first1)),
        "followed_to_28d": int(((all_first1.last_age >= 28)).sum()),
        "followed_to_28d_and_15obs": int(((all_first1.last_age >= 28) & (all_first1.n_obs >= 15)).sum()),
    }
    cohort_ids = per[(per.first_age <= 1.0) & (per.last_age >= 28) & (per.n_obs >= 15)].index

    # ---- trajectories (interpolate on monotone cumulative views, log scale)
    log("interpolating trajectories ...")
    sub = df[df.video_id.isin(cohort_ids)].sort_values(["video_id", "age_days"])
    rows, pl_rows, reacc_rows = [], [], []
    for vid, grp in sub.groupby("video_id", sort=False):
        a = grp.age_days.to_numpy(); v = np.maximum.accumulate(grp.views.to_numpy(dtype=float))
        lv = np.log1p(v)
        inter = np.interp(AGES, a, lv, left=np.nan, right=np.nan)
        lin = np.interp([1, 2, 3, 7, 28], a, v, left=np.nan, right=np.nan)
        def _next(t):
            i = np.searchsorted(a, t); return v[i] if i < len(v) else np.nan
        rows.append([vid] + list(np.expm1(inter)) + list(lin) + [_next(1), _next(3)])
        m = (a >= 0.5) & (v > 0)
        if m.sum() >= 8:
            pl_rows.append((vid, np.polyfit(np.log(a[m]), np.log(v[m]), 1)[0]))
        inc = np.diff(v) / np.maximum(np.diff(a), 1e-6); mid = a[1:]
        early = inc[(mid >= 1) & (mid <= 7)]; late = inc[mid > 7]
        if len(early) >= 3 and len(late) >= 10:
            reacc_rows.append((vid, bool(late.max() > early.max())))
    traj = pd.DataFrame(rows, columns=["video_id"] + [f"V{a}" for a in AGES] + ["V1_lin", "V2_lin", "V3_lin", "V7_lin", "V28_lin", "V1_next", "V3_next"]).set_index("video_id")
    traj = traj.join(per[["category", "channel", "dur", "pub", "ch_country", "seg", "lang", "order", "subs"]])
    traj = traj.join(pd.DataFrame(pl_rows, columns=["video_id", "pl_b"]).set_index("video_id"))
    traj = traj.join(pd.DataFrame(reacc_rows, columns=["video_id", "reacc"]).set_index("video_id"))
    traj = traj.dropna(subset=["V1", "V28"])
    traj = traj[traj.V28 > 0]
    R["cohort_funnel"]["usable_trajectories"] = int(len(traj))
    cat_n = traj.category.value_counts()
    keep_cats = cat_n[cat_n >= CAT_MIN_N].index.tolist()
    R["cohort_funnel"]["categories_kept"] = {c: int(cat_n[c]) for c in keep_cats}
    R["cohort_funnel"]["categories_excluded"] = {c: int(cat_n[c]) for c in cat_n.index if c not in keep_cats}
    traj = traj[traj.category.isin(keep_cats)].copy()
    R["cohort_funnel"]["analytic_N"] = int(len(traj))
    R["cohort_funnel"]["analytic_channels"] = int(traj.channel.nunique())
    for k in ("V1", "V3", "V7", "V14"):
        traj[f"s{k}"] = traj[k] / traj.V28
    traj["late_ratio"] = traj.V28 / traj.V7.clip(lower=1)
    traj["late2"] = (traj.late_ratio >= 2).astype(float)
    traj["late3"] = (traj.late_ratio >= 3).astype(float)
    traj["logV28"] = np.log10(traj.V28.clip(lower=1))
    traj["dur_bin"] = pd.cut(traj.dur, [0, 15, 30, 45, 60, 1e9], labels=["<=15 s", "16-30 s", "31-45 s", "46-60 s", ">60 s"])
    traj["V7q"] = pd.qcut(np.log10(traj.V7.clip(lower=1)), 5, labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
    traj["V28q"] = pd.qcut(traj.logV28, 5, labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
    traj["order"] = traj.order.fillna("unknown")

    # ---- Table 1 description
    R["sample"] = {
        "N": int(len(traj)), "channels": int(traj.channel.nunique()),
        "videos_per_channel_q": traj.groupby("channel").size().quantile([.5, .75, .9, .99]).to_dict(),
        "category_counts": traj.category.value_counts().to_dict(),
        "segment_counts": traj.seg.value_counts().reindex(SEG_ORDER).fillna(0).astype(int).to_dict(),
        "ch_country_top": traj.ch_country.fillna("NA").value_counts().head(10).to_dict(),
        "order_counts": traj.order.value_counts().to_dict(),
        "duration_q": traj.dur.quantile([.1, .5, .9]).to_dict(), "duration_le60_share": float((traj.dur <= 60).mean()),
        "subs_q": traj.subs.quantile([.1, .25, .5, .75, .9]).to_dict(),
        "V28_q": traj.V28.quantile([.1, .25, .5, .75, .9, .99]).round(0).to_dict(),
        "V28_top1pct_share": float(traj.V28.sort_values(ascending=False).head(int(len(traj) * .01)).sum() / traj.V28.sum()),
        "pub_range": [str(traj.pub.min().date()), str(traj.pub.max().date())],
        "ja_audio_share": float(traj.lang.fillna("").str.startswith("ja").mean()),
        "obs_per_video_q": per.loc[traj.index, "n_obs"].quantile([.1, .5, .9]).to_dict(),
    }

    # ---- lifecycle shape (Table 2)
    log("bootstrapping lifecycle shape ...")
    shape = {}
    for k in ("sV1", "sV3", "sV7", "sV14"):
        shape[k] = boot_table(traj, None, k, nanmedian, B, rng).iloc[0].to_dict()
    shape["pl_b"] = boot_table(traj.dropna(subset=["pl_b"]), None, "pl_b", nanmedian, B, rng).iloc[0].to_dict()
    R["shape_overall"] = shape
    R["shape_by_category"] = {k: boot_table(traj, "category", k, nanmedian, B, rng).to_dict("records") for k in ("sV1", "sV3", "sV7")}
    R["shape_by_category"]["medV28"] = boot_table(traj, "category", "V28", nanmedian, B, rng).to_dict("records")
    R["shape_by_category"]["pl_b"] = boot_table(traj.dropna(subset=["pl_b"]), "category", "pl_b", nanmedian, B, rng).to_dict("records")
    R["shape_by_dur"] = {k: boot_table(traj, "dur_bin", k, nanmedian, B, rng).to_dict("records") for k in ("sV3", "V28")}
    R["shape_vs_reach"] = {
        "spearman_sV3_logV28": float(stats.spearmanr(traj.sV3, traj.logV28).correlation),
        "spearman_sV3_logsubs": float(stats.spearmanr(traj.sV3[traj.subs > 0], np.log10(traj.subs[traj.subs > 0])).correlation),
        "spearman_logsubs_logV28": float(stats.spearmanr(np.log10(traj.subs[traj.subs > 0]), traj.logV28[traj.subs > 0]).correlation),
    }
    # median normalized curve by age (for Fig 1): per age, median of V_t/V28, IQR
    curve = {}
    for a in AGES:
        s = traj[f"V{a}"] / traj.V28
        curve[a] = {"q25": float(s.quantile(.25)), "q50": float(s.median()), "q75": float(s.quantile(.75))}
    R["curve_overall"] = curve
    R["curve_by_category"] = {c: {a: float((g[f"V{a}"] / g.V28).median()) for a in AGES} for c, g in traj.groupby("category")}

    # ---- predictability (Table 3)
    log("bootstrapping predictability ...")
    def fit_stats(d: pd.DataFrame, k: str):
        x = np.log10(d[k].clip(lower=1)); y = d.logV28
        res = stats.linregress(x, y)
        return {"R2": res.rvalue ** 2, "slope": res.slope, "intercept": res.intercept, "spearman": stats.spearmanr(x, y).correlation}
    pred = {}
    for k in ("V1", "V2", "V3", "V7", "V14"):
        pt = fit_stats(traj, k)
        # cluster bootstrap of R2 and slope
        ch = traj.channel.to_numpy(); uniq, inv = np.unique(ch, return_inverse=True)
        idx_by_c = [[] for _ in uniq]
        for i, c in enumerate(inv): idx_by_c[c].append(i)
        idx_by_c = [np.asarray(x) for x in idx_by_c]
        x_all = np.log10(traj[k].clip(lower=1).to_numpy()); y_all = traj.logV28.to_numpy()
        r2s, sls = np.empty(B), np.empty(B)
        for b in range(B):
            sel = np.concatenate([idx_by_c[c] for c in rng.integers(0, len(uniq), len(uniq))])
            res = stats.linregress(x_all[sel], y_all[sel]); r2s[b] = res.rvalue ** 2; sls[b] = res.slope
        pt.update({"R2_lo": float(np.percentile(r2s, 2.5)), "R2_hi": float(np.percentile(r2s, 97.5)),
                   "slope_lo": float(np.percentile(sls, 2.5)), "slope_hi": float(np.percentile(sls, 97.5))})
        pred[k] = pt
    R["predictability"] = pred
    # top-decile persistence d2 -> d28 and d3 -> d28
    pers = {}
    for k in ("V1", "V2", "V3", "V7"):
        topk = traj[k] >= traj[k].quantile(.9); top28 = traj.V28 >= traj.V28.quantile(.9)
        pers[k] = {"P_top28_given_topk": float((topk & top28).sum() / topk.sum()), "share_top28_not_topk": float(1 - (topk & top28).sum() / top28.sum())}
    R["decile_persistence"] = pers
    R["predictability_by_order"] = {o: fit_stats(d, "V3") | {"n": int(len(d))} for o, d in traj.groupby("order") if len(d) >= 500}
    R["predictability_by_category"] = {c: fit_stats(d, "V3") | {"n": int(len(d))} for c, d in traj.groupby("category")}
    R["predictability_by_segment"] = {s: fit_stats(d, "V3") | {"n": int(len(d))} for s, d in traj.groupby("seg")}

    # ---- late growth (Table 4)
    log("bootstrapping late growth ...")
    R["late_overall"] = {"late2": boot_table(traj, None, "late2", mean_, B, rng).iloc[0].to_dict(),
                         "late3": boot_table(traj, None, "late3", mean_, B, rng).iloc[0].to_dict(),
                         "reacc": boot_table(traj.dropna(subset=["reacc"]).assign(reacc=lambda d: d.reacc.astype(float)), None, "reacc", mean_, B, rng).iloc[0].to_dict(),
                         "late_ratio_median": boot_table(traj, None, "late_ratio", nanmedian, B, rng).iloc[0].to_dict()}
    R["late_by_V7q"] = boot_table(traj, "V7q", "late2", mean_, B, rng).to_dict("records")
    R["late_by_V7q_medV28"] = traj.groupby("V7q", observed=True).V28.median().to_dict()
    R["late_by_category"] = boot_table(traj, "category", "late2", mean_, B, rng).to_dict("records")
    R["late_by_segment"] = boot_table(traj, "seg", "late2", mean_, B, rng, order=SEG_ORDER).to_dict("records")
    lb = traj[traj.late2 == 1]; nb = traj[traj.late2 == 0]
    R["late_profile"] = {"n_late": int(len(lb)), "medV7_late": float(lb.V7.median()), "medV7_other": float(nb.V7.median()),
                         "medV28_late": float(lb.V28.median()), "medV28_other": float(nb.V28.median())}

    # ---- heterogeneity by creator segment (Table 5)
    log("bootstrapping segments ...")
    R["seg_table"] = {k: boot_table(traj, "seg", k, nanmedian, B, rng, order=SEG_ORDER).to_dict("records") for k in ("V28", "sV1", "sV3", "sV7")}
    R["seg_table"]["subs_median"] = traj.groupby("seg").subs.median().reindex(SEG_ORDER).to_dict()
    R["seg_table"]["V28_over_subs_median"] = traj[traj.subs > 0].assign(r=lambda d: d.V28 / d.subs).groupby("seg").r.median().reindex(SEG_ORDER).to_dict()
    R["seg_x_category_sV3"] = pd.pivot_table(traj[traj.seg.isin(["JP", "US", "IN"])], index="category", columns="seg", values="sV3", aggfunc="median").round(3).to_dict()
    R["seg_x_category_medV28"] = pd.pivot_table(traj[traj.seg.isin(["JP", "US", "IN"])], index="category", columns="seg", values="V28", aggfunc="median").round(0).to_dict()
    R["seg_x_category_n"] = pd.crosstab(traj.category, traj.seg).to_dict()

    # ---- within-channel decomposition
    multi = traj.groupby("channel").filter(lambda d: len(d) >= 3).copy()
    dec = {"channels": int(multi.channel.nunique()), "videos": int(len(multi))}
    for col in ("logV28", "sV3", "late_ratio"):
        y = multi[col]; within = (y - y.groupby(multi.channel).transform("mean")).var(); dec[f"within_share_{col}"] = float(within / y.var())
    multi["logV28_dm"] = multi.logV28 - multi.groupby("channel").logV28.transform("mean")
    multi["sV3_dm"] = multi.sV3 - multi.groupby("channel").sV3.transform("mean")
    dec["dur_bin_logV28_dm"] = multi.groupby("dur_bin", observed=True).logV28_dm.agg(["mean", "size"]).to_dict()
    dec["dur_bin_sV3_dm"] = multi.groupby("dur_bin", observed=True).sV3_dm.agg(["mean", "size"]).to_dict()
    R["within_channel"] = dec

    # ---- sensitivity: seed order, publication week
    R["sens_order"] = {}
    for o, d in traj.groupby("order"):
        if len(d) < 500: continue
        R["sens_order"][o] = {"n": int(len(d)), "medV28": float(d.V28.median()), "sV1": float(d.sV1.median()), "sV3": float(d.sV3.median()),
                              "sV7": float(d.sV7.median()), "late2": float(d.late2.mean()), "R2_V3": fit_stats(d, "V3")["R2"], "slope_V3": fit_stats(d, "V3")["slope"]}
    traj["pubweek"] = traj.pub.dt.strftime("%G-W%V")
    R["by_pubweek"] = traj.groupby("pubweek").agg(n=("V28", "size"), medV3=("V3", "median"), medV28=("V28", "median"), sV3=("sV3", "median"), late2=("late2", "mean")).round(3).to_dict("index")

    # ---- reviewer-driven robustness block ----
    log("robustness: increment model, interpolation, size layers, thresholds ...")
    ch_all = traj.channel.to_numpy(); uniq_all, inv_all = np.unique(ch_all, return_inverse=True)
    idx_all = [[] for _ in uniq_all]
    for i, c in enumerate(inv_all): idx_all[c].append(i)
    idx_all = [np.asarray(x) for x in idx_all]

    def boot_fit(x_all, y_all, mask=None):
        m = np.ones(len(x_all), bool) if mask is None else mask
        res = stats.linregress(x_all[m], y_all[m])
        r2s, sls = np.empty(B), np.empty(B)
        for b in range(B):
            sel = np.concatenate([idx_all[c] for c in rng.integers(0, len(uniq_all), len(uniq_all))])
            sel = sel[m[sel]]
            rr = stats.linregress(x_all[sel], y_all[sel]); r2s[b] = rr.rvalue ** 2; sls[b] = rr.slope
        return {"R2": res.rvalue ** 2, "slope": res.slope, "intercept": res.intercept, "n": int(m.sum()),
                "R2_lo": float(np.percentile(r2s, 2.5)), "R2_hi": float(np.percentile(r2s, 97.5)),
                "slope_lo": float(np.percentile(sls, 2.5)), "slope_hi": float(np.percentile(sls, 97.5))}

    def cluster_boot_df(d: pd.DataFrame, stat_fn, nb=None):
        nb = nb or B
        ch = d["channel"].to_numpy(); uq, iv = np.unique(ch, return_inverse=True)
        ib = [[] for _ in uq]
        for i, c in enumerate(iv): ib[c].append(i)
        ib = [np.asarray(x) for x in ib]
        point = float(stat_fn(d)); out = np.empty(nb)
        for b in range(nb):
            sel = np.concatenate([ib[c] for c in rng.integers(0, len(uq), len(uq))]); out[b] = stat_fn(d.iloc[sel])
        return {"est": point, "lo": float(np.nanpercentile(out, 2.5)), "hi": float(np.nanpercentile(out, 97.5)), "n": int(len(d)), "n_channels": int(len(uq))}

    # (1) increment model: growth after day t, and ratio distributions
    inc = {}
    y28 = traj.logV28.to_numpy()
    for k in ("V1", "V2", "V3", "V7"):
        gain = (traj.V28 - traj[k]).clip(lower=0).to_numpy(); mask = gain > 0
        xk = np.log10(traj[k].clip(lower=1).to_numpy()); yg = np.log10(gain + 1)
        inc[k] = boot_fit(xk, yg, mask)
        ratio = traj.V28 / traj[k].clip(lower=1)
        inc[k]["ratio_q"] = ratio.quantile([.1, .25, .5, .75, .9, .99]).round(3).to_dict()
        inc[k]["spearman_ratio_vs_Vt"] = float(stats.spearmanr(ratio, traj[k]).correlation)
    R["increment_model"] = inc
    # Szabo-Huberman aligned: V7 -> V28 nested fit already in R["predictability"]["V7"]

    # (2) interpolation sensitivity
    sens = {}
    for t in ("1", "2", "3", "7"):
        sens[f"S{t}_loglinear"] = float((traj[f"V{t}"] / traj.V28).median())
        sens[f"S{t}_linear"] = float((traj[f"V{t}_lin"] / traj.V28_lin).median())
    sens["S1_next_obs"] = float((traj.V1_next / traj.V28).median()); sens["S3_next_obs"] = float((traj.V3_next / traj.V28).median())
    fa = per.loc[traj.index, "first_age"]
    sens["first_age_q"] = fa.quantile([.1, .25, .5, .75, .9]).round(3).to_dict()
    sens["share_first_le_0.25d"] = float((fa <= .25).mean()); sens["share_first_le_0.5d"] = float((fa <= .5).mean())
    # S1 restricted to videos first observed within 6 h (interpolation span short)
    early = traj[fa.reindex(traj.index) <= .25]
    sens["S1_loglinear_first_le_6h"] = float((early.V1 / early.V28).median()); sens["S1_linear_first_le_6h"] = float((early.V1_lin / early.V28_lin).median()); sens["n_first_le_6h"] = int(len(early))
    R["interp_sensitivity"] = sens

    # (3) channel-size layer
    traj["subq"] = pd.qcut(np.log10(traj.subs.clip(lower=1)), 5, labels=["S1 (smallest)", "S2", "S3", "S4", "S5 (largest)"])
    R["subs_quintile_bounds"] = traj.groupby("subq", observed=True).subs.agg(["min", "max", "median"]).to_dict("index")
    R["sV3_by_subq"] = boot_table(traj, "subq", "sV3", nanmedian, B, rng).to_dict("records")
    R["sV7_by_subq"] = boot_table(traj, "subq", "sV7", nanmedian, B, rng).to_dict("records")
    R["late2_by_subq"] = boot_table(traj, "subq", "late2", mean_, B, rng).to_dict("records")
    R["medV28_by_subq"] = traj.groupby("subq", observed=True).V28.median().to_dict()
    R["sV3_subq_x_seg"] = pd.pivot_table(traj[traj.seg.isin(["JP", "US", "IN"])], index="subq", columns="seg", values="sV3", aggfunc="median", observed=True).round(3).to_dict()
    R["n_subq_x_seg"] = pd.crosstab(traj.subq, traj.seg).to_dict()
    pv = pd.pivot_table(traj, index="category", columns="subq", values="sV3", aggfunc="median", observed=True)
    wq = traj.subq.value_counts(normalize=True).reindex(pv.columns).to_numpy()
    R["sV3_category_x_subq"] = pv.round(3).to_dict()
    R["sV3_category_size_standardized"] = {c: float(np.nansum(pv.loc[c].to_numpy() * wq) / np.nansum(wq[~np.isnan(pv.loc[c].to_numpy())])) for c in pv.index}
    R["subs_median_by_category"] = traj.groupby("category").subs.median().to_dict()
    # segment sV3 standardized to the overall category mix (direct standardization) with cluster CI
    wc = traj.category.value_counts(normalize=True)
    def std_fn(d):
        med = d.groupby("category").sV3.median(); cats = [c for c in wc.index if c in med.index]
        return sum(wc[c] * med[c] for c in cats) / sum(wc[c] for c in cats)
    R["sV3_seg_category_standardized"] = {s: cluster_boot_df(traj[traj.seg == s], std_fn, nb=min(B, 500)) for s in SEG_ORDER}
    # segment sV3 standardized to overall subs-quintile mix
    wsq = traj.subq.value_counts(normalize=True)
    def std_sq(d):
        med = d.groupby("subq", observed=True).sV3.median(); qs = [q for q in wsq.index if q in med.index]
        return sum(wsq[q] * med[q] for q in qs) / sum(wsq[q] for q in qs)
    R["sV3_seg_size_standardized"] = {s: cluster_boot_df(traj[traj.seg == s], std_sq, nb=min(B, 500)) for s in SEG_ORDER}

    # (4) late-growth threshold sensitivity and absolute-increment view
    traj["late15"] = (traj.late_ratio >= 1.5).astype(float)
    R["late_thresholds"] = {th: boot_table(traj, None, col, mean_, B, rng).iloc[0].to_dict() for th, col in (("1.5x", "late15"), ("2x", "late2"), ("3x", "late3"))}
    R["late15_by_V7q"] = boot_table(traj, "V7q", "late15", mean_, B, rng).to_dict("records")
    R["late3_by_V7q"] = boot_table(traj, "V7q", "late3", mean_, B, rng).to_dict("records")
    gain7 = (traj.V28 - traj.V7).clip(lower=0); top_gain = gain7 >= gain7.quantile(.9)
    R["top_decile_absolute_gain_by_V7q"] = traj.assign(tg=top_gain.astype(float)).groupby("V7q", observed=True).tg.mean().round(4).to_dict()
    R["late2_by_subq_x_V7q"] = pd.pivot_table(traj, index="subq", columns="V7q", values="late2", aggfunc="mean", observed=True).round(4).to_dict()

    # (5) survivorship: videos lost before day 28 are absent from the lifecycle cohort
    Ls = led.copy()
    for c in ("published_at", "last_observed_at"):
        Ls[c] = pd.to_datetime(Ls[c], utc=True, errors="coerce")
    Ls["last_age"] = (Ls.last_observed_at - Ls.published_at).dt.total_seconds() / 86400
    end_ts = df.observed_at.max()
    elig28 = Ls[(Ls.seed_age_days <= 1.0) & ((end_ts - Ls.published_at).dt.total_seconds() / 86400 >= 28)]
    lost = elig28[(elig28.retired_reason == "missing") & (elig28.last_age < 28)]
    surv = {"eligible_28d": int(len(elig28)), "lost_before_28d": int(len(lost)), "share_lost": float(len(lost) / max(len(elig28), 1))}
    # early views of lost videos vs cohort (V3 interpolated where available)
    sub_l = df[df.video_id.isin(lost.index)].sort_values(["video_id", "age_days"])
    v3l = []
    for vid, g in sub_l.groupby("video_id", sort=False):
        a = g.age_days.to_numpy(); v = np.maximum.accumulate(g.views.to_numpy(dtype=float))
        if a.min() <= 1 and a.max() >= 3:
            v3l.append(float(np.expm1(np.interp(3, a, np.log1p(v)))))
    surv["lost_with_V3"] = len(v3l); surv["medV3_lost"] = float(np.median(v3l)) if v3l else None; surv["medV3_cohort"] = float(traj.V3.median())
    surv["lost_last_age_q"] = lost.last_age.quantile([.1, .5, .9]).round(2).to_dict()
    R["survivorship"] = surv

    # (6) Kaplan-Meier cumulative disappearance (seeded within 24 h), overall and by segment
    km_src = Ls[Ls.seed_age_days <= 1.0].copy()
    km_src = km_src.join(meta[["ch_country"]], how="left", rsuffix="_m"); km_src["seg"] = segment(km_src.ch_country)
    km_src["event"] = (km_src.retired_reason == "missing").astype(int)
    km_src["time"] = np.where(km_src.event == 1, km_src.last_age, np.minimum(km_src.last_age.fillna(0), 35.0))
    km_src = km_src.dropna(subset=["time"]); km_src = km_src[km_src.time >= 0]

    def km(times, events, grid):
        times = np.asarray(times, float); events = np.asarray(events, int)
        ev_t = np.unique(times[events == 1]); S = 1.0; keys, vals = [], []
        for ut in ev_t:
            at_risk = int(np.sum(times >= ut)); d = int(np.sum((times == ut) & (events == 1)))
            if at_risk > 0: S *= (1 - d / at_risk)
            keys.append(ut); vals.append(S)
        keys = np.array(keys); vals = np.array(vals); out = []
        for g in grid:
            k = np.searchsorted(keys, g, side="right"); out.append(1.0 if k == 0 else float(vals[k - 1]))
        return [1 - x for x in out]
    grid = list(range(0, 36))
    R["km_grid"] = grid
    R["km_overall"] = km(km_src.time, km_src.event, grid)
    R["km_by_segment"] = {s: km(km_src[km_src.seg == s].time, km_src[km_src.seg == s].event, grid) for s in SEG_ORDER}
    R["km_n"] = {"all": int(len(km_src)), **{s: int((km_src.seg == s).sum()) for s in SEG_ORDER}, "events": int(km_src.event.sum())}
    R["km_by_category"] = {c: km(km_src[km_src.category_name == c].time, km_src[km_src.category_name == c].event, grid) for c in keep_cats}

    # (7) channel-level view of Music disappearance (robustness to a few prolific channels)
    elapsed35 = (end_ts - Ls.published_at).dt.total_seconds() / 86400 >= 35
    done_m = Ls[(Ls.seed_age_days <= 1.0) & elapsed35 & (Ls.category_name == "Music")].copy()
    done_m["miss"] = (done_m.retired_reason == "missing").astype(float)
    chm = done_m.groupby("channel_id").agg(n=("miss", "size"), k=("miss", "sum"))
    capped = done_m.groupby("channel_id", group_keys=False).apply(lambda d: d.sample(min(len(d), 3), random_state=0))
    R["music_channel_view"] = {"videos": int(len(done_m)), "channels": int(len(chm)), "share_channels_with_any_missing": float((chm.k > 0).mean()),
                               "channel_mean_rate": float((chm.k / chm.n).mean()), "capped3_rate": float(capped.miss.mean()), "capped3_n": int(len(capped)),
                               "top_channel_share_of_missing": float(chm.k.max() / max(chm.k.sum(), 1)), "top5_channels_share_of_missing": float(chm.k.sort_values(ascending=False).head(5).sum() / max(chm.k.sum(), 1))}
    # same for all categories combined
    done_all = Ls[(Ls.seed_age_days <= 1.0) & elapsed35].copy(); done_all["miss"] = (done_all.retired_reason == "missing").astype(float)
    cha = done_all.groupby("channel_id").agg(n=("miss", "size"), k=("miss", "sum"))
    R["all_channel_view"] = {"channels": int(len(cha)), "share_channels_with_any_missing": float((cha.k > 0).mean()), "channel_mean_rate": float((cha.k / cha.n).mean()),
                             "capped3_rate": float(done_all.groupby("channel_id", group_keys=False).apply(lambda d: d.sample(min(len(d), 3), random_state=0)).miss.mean()),
                             "top5_channels_share_of_missing": float(cha.k.sort_values(ascending=False).head(5).sum() / max(cha.k.sum(), 1))}

    # (7b) calendar-cohort heterogeneity of disappearance (seed month) and the most prolific Music channel
    Ls["seed_month"] = Ls.seed_source.astype(str).str.extract(r"daily_(\d{6})")[0]
    cohort_rows = {}
    for mth, g in km_src.join(Ls[["seed_month"]], how="left").groupby("seed_month"):
        comp = g[(end_ts - g.published_at).dt.total_seconds() / 86400 >= 35]
        cohort_rows[mth] = {"seeds": int(len(g)), "completed": int(len(comp)), "crude_rate_completed": float((comp.retired_reason == "missing").mean()) if len(comp) else None,
                            "km35": float(km(g.time, g.event, [35])[0]), "km7": float(km(g.time, g.event, [7])[0]), "events": int(g.event.sum())}
    R["disappearance_by_seed_month"] = cohort_rows
    topc = chm.k.idxmax()
    tc = done_m[done_m.channel_id == topc]
    R["music_top_channel"] = {"videos_in_music_done": int(len(tc)), "missing": int(tc.miss.sum()),
                              "last_obs_range": [str(tc.last_observed_at.min().date()), str(tc.last_observed_at.max().date())],
                              "seed_months": Ls.loc[tc.index, "seed_month"].value_counts().to_dict(),
                              "ch_country": str(meta.ch_country.reindex(tc.index).mode().iloc[0]) if len(tc) else None}
    # Music disappearance excluding the single most prolific channel
    R["music_rate_excl_top_channel"] = float(done_m[done_m.channel_id != topc].miss.mean())
    R["disappearance_rate_excl_top5_channels"] = float(done_all[~done_all.channel_id.isin(cha.k.sort_values(ascending=False).head(5).index)].miss.mean())

    # (8) hidden like counts: selectivity check
    dh = df[df.video_id.isin(traj.index) & (df.views > 1000)].copy()
    hid = dh.assign(h=(dh.likes == 0).astype(float)).groupby("video_id").h.mean()
    traj["likes_hidden"] = (hid.reindex(traj.index) >= 0.5)
    hv = traj[traj.likes_hidden == True]; sv = traj[traj.likes_hidden == False]
    R["hidden_likes"] = {"share_videos_hidden": float(traj.likes_hidden.mean()), "n_hidden": int(len(hv)),
                         "by_segment": traj.groupby("seg").likes_hidden.mean().reindex(SEG_ORDER).round(4).to_dict(),
                         "by_category": traj.groupby("category").likes_hidden.mean().round(4).to_dict(),
                         "medV28_hidden": float(hv.V28.median()), "medV28_shown": float(sv.V28.median()),
                         "sV3_hidden": float(hv.sV3.median()), "sV3_shown": float(sv.sV3.median()),
                         "late2_hidden": float(hv.late2.mean()), "late2_shown": float(sv.late2.mean())}

    # ---- engagement dilution (Table 6)
    log("engagement dilution ...")
    d = df[(df.views > 1000) & (df.likes > 0)].copy(); d["lr"] = d.likes / d.views
    early = d[d.age_days <= 1.5].sort_values("age_days").groupby("video_id").lr.first()
    late = d[d.age_days >= 25].sort_values("age_days").groupby("video_id").lr.last()
    both = pd.concat([early.rename("e"), late.rename("l")], axis=1).dropna()
    both = both.join(traj[["channel", "seg", "category", "V28q", "V28"]], how="inner")
    both["ratio"] = both.l / both.e; both["declined"] = (both.l < both.e).astype(float)
    R["like_dilution"] = {
        "n": int(len(both)), "early_median": float(both.e.median()), "late_median": float(both.l.median()),
        "ratio": boot_table(both, None, "ratio", nanmedian, B, rng).iloc[0].to_dict(),
        "share_declined": boot_table(both, None, "declined", mean_, B, rng).iloc[0].to_dict(),
        "ratio_by_V28q": boot_table(both, "V28q", "ratio", nanmedian, B, rng).to_dict("records"),
        "ratio_by_seg": boot_table(both, "seg", "ratio", nanmedian, B, rng, order=SEG_ORDER).to_dict("records"),
        "likes_hidden_row_share": float(((df.likes == 0) & (df.views > 1000)).mean()),
    }
    # like rate by age bin x reach quintile (Fig 5)
    dd = df[df.video_id.isin(traj.index) & (df.views > 1000) & (df.likes > 0)].copy()
    dd["lr"] = dd.likes / dd.views; dd["V28q"] = traj.V28q.reindex(dd.video_id).to_numpy()
    dd["age_bin"] = pd.cut(dd.age_days, [0, 1, 2, 3, 5, 7, 10, 14, 21, 28, 35], labels=[1, 2, 3, 5, 7, 10, 14, 21, 28, 35])
    lr_curve = dd.groupby(["V28q", "age_bin"], observed=True).lr.median().unstack("V28q")
    R["like_rate_curve"] = {str(c): {str(i): float(v) for i, v in lr_curve[c].items()} for c in lr_curve.columns}
    # view decreases
    s2 = df.sort_values(["video_id", "age_days"]); dv = s2.groupby("video_id").views.diff()
    R["view_decreases"] = {"pairs": int(dv.notna().sum()), "decreases": int((dv < 0).sum()), "share_pairs": float((dv < 0).sum() / dv.notna().sum()),
                           "videos_any": int(s2[dv < 0].video_id.nunique()), "videos_share": float(s2[dv < 0].video_id.nunique() / df.video_id.nunique()),
                           "magnitude_q": (-dv[dv < 0]).quantile([.5, .9, .99]).to_dict()}

    # ---- disappearance (Table 7)
    log("disappearance ...")
    L = led.copy()
    for c in ("published_at", "first_observed_at", "last_observed_at"):
        L[c] = pd.to_datetime(L[c], utc=True, errors="coerce")
    L = L.join(meta[["ch_country", "title"]], how="left"); L["seg"] = segment(L.ch_country)
    L["channel"] = L.channel_id
    L["last_age"] = (L.last_observed_at - L.published_at).dt.total_seconds() / 86400
    end = df.observed_at.max()
    seeded = L[L.seed_age_days <= 1.0]
    # 35-day cohort = seeds whose full 35-day window has elapsed by the last run (NOT "retired" status, which would
    # include early disappearances of recent cohorts without their surviving peers)
    done = seeded[(end - seeded.published_at).dt.total_seconds() / 86400 >= 35].copy(); done["miss"] = (done.retired_reason == "missing").astype(float)
    elig7 = seeded[(end - seeded.published_at).dt.total_seconds() / 86400 >= 7].copy()
    elig7["miss7"] = ((elig7.retired_reason == "missing") & (elig7.last_age <= 7)).astype(float)
    lastv = df.sort_values("age_days").groupby("video_id").views.last()
    done["lastv"] = lastv.reindex(done.index); done["dur"] = df.groupby("video_id").duration_s.first().reindex(done.index)
    done["vbin"] = pd.cut(done.lastv, [-1, 1e3, 1e4, 1e5, 1e6, 1e12], labels=["<1k", "1k-10k", "10k-100k", "100k-1M", ">1M"])
    done["dur_bin"] = pd.cut(done.dur, [0, 15, 30, 45, 60, 1e9], labels=["<=15 s", "16-30 s", "31-45 s", "46-60 s", ">60 s"])
    done_cat = done[done.category_name.isin(keep_cats)]
    miss = done[done.miss == 1]
    R["disappearance"] = {
        "seeded_within_24h": int(len(seeded)), "completed_35d_window": int(len(done)), "eligible_7d": int(len(elig7)),
        "rate_35d": boot_table(done, None, "miss", mean_, B, rng).iloc[0].to_dict(),
        "rate_7d": boot_table(elig7, None, "miss7", mean_, B, rng).iloc[0].to_dict(),
        "by_segment": boot_table(done, "seg", "miss", mean_, B, rng, order=SEG_ORDER).to_dict("records"),
        "by_category": boot_table(done_cat, "category_name", "miss", mean_, B, rng).to_dict("records"),
        "by_country_top": boot_table(done.assign(cc=done.ch_country.fillna("NA")), "cc", "miss", mean_, B, rng, min_n=300).sort_values("est", ascending=False).to_dict("records"),
        "by_vbin": boot_table(done.dropna(subset=["vbin"]), "vbin", "miss", mean_, B, rng).to_dict("records"),
        "by_dur": boot_table(done.dropna(subset=["dur_bin"]), "dur_bin", "miss", mean_, B, rng).to_dict("records"),
        "seg_x_category": pd.pivot_table(done_cat[done_cat.seg.isin(["JP", "US", "IN"])], index="category_name", columns="seg", values="miss", aggfunc="mean").round(3).to_dict(),
        "missing_n": int(len(miss)), "missing_age_q": miss.last_age.quantile([.1, .5, .9]).round(1).to_dict(),
        "missing_lastviews_q": miss.lastv.quantile([.1, .5, .9]).to_dict(), "missing_lastviews_sum": float(miss.lastv.sum()),
        "status_paths": df.sort_values("age_days").groupby("video_id").privacy_status.agg(lambda s: "->".join(dict.fromkeys(s))).value_counts().to_dict(),
    }
    # Music keyword check (title) within done
    mu = done[done.category_name == "Music"].copy(); low = mu.title.fillna("").str.lower()
    kw = {}
    for w in ("song", "lyrics", "cover", "status", "whatsapp", "trending", "viral"):
        k = low.str.contains(w, regex=False)
        if k.sum() >= 50: kw[w] = {"n": int(k.sum()), "miss_with": float(mu[k].miss.mean()), "miss_without": float(mu[~k].miss.mean())}
    R["disappearance"]["music_keywords"] = kw

    # ---- save JSON
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (pd.Timestamp,)): return str(o)
        if isinstance(o, (pd.Interval,)): return str(o)
        return str(o)
    stem.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1, default=conv)
    traj.drop(columns=["pub"]).to_parquet(f"{stem}_traj.parquet")
    log("saved", f"{stem}.json")

    # ---- markdown tables
    md = []
    S = R["sample"]
    md.append(f"## Sample\nN={S['N']} videos, {S['channels']} channels; pub {S['pub_range']}; median duration {S['duration_q'][0.5]:.0f}s; median V28 {S['V28_q'][0.5]:,.0f}; top1% share {100*S['V28_top1pct_share']:.1f}%\n")
    md.append("| Category | n | median V28 | share d1 | share d3 | share d7 | late>=2x |\n|---|---|---|---|---|---|---|")
    cat_rows = {r["group"]: r for r in R["shape_by_category"]["sV3"]}
    for c in sorted(cat_rows, key=lambda c: cat_rows[c]["est"]):
        s1 = next(r for r in R["shape_by_category"]["sV1"] if r["group"] == c); s3 = cat_rows[c]; s7 = next(r for r in R["shape_by_category"]["sV7"] if r["group"] == c)
        v = next(r for r in R["shape_by_category"]["medV28"] if r["group"] == c); l2 = next(r for r in R["late_by_category"] if r["group"] == c)
        md.append(f"| {c} | {s3['n']} | {v['est']:,.0f} | {fmt_pct(s1['est'], s1['lo'], s1['hi'])} | {fmt_pct(s3['est'], s3['lo'], s3['hi'])} | {fmt_pct(s7['est'], s7['lo'], s7['hi'])} | {fmt_pct(l2['est'], l2['lo'], l2['hi'])} |")
    md.append("\n| Predictor | R2 [CI] | slope [CI] | Spearman |\n|---|---|---|---|")
    for k, p in R["predictability"].items():
        md.append(f"| log {k} | {fmt_num(p['R2'], p['R2_lo'], p['R2_hi'], 3)} | {fmt_num(p['slope'], p['slope_lo'], p['slope_hi'], 3)} | {p['spearman']:.3f} |")
    md.append("\n| Segment | n | median V28 | share d1 | share d3 | share d7 | late>=2x | 35d disappearance |\n|---|---|---|---|---|---|---|---|")
    for s in SEG_ORDER:
        v = next(r for r in R["seg_table"]["V28"] if r["group"] == s); s1 = next(r for r in R["seg_table"]["sV1"] if r["group"] == s)
        s3 = next(r for r in R["seg_table"]["sV3"] if r["group"] == s); s7 = next(r for r in R["seg_table"]["sV7"] if r["group"] == s)
        l2 = next(r for r in R["late_by_segment"] if r["group"] == s); dis = next(r for r in R["disappearance"]["by_segment"] if r["group"] == s)
        md.append(f"| {SEG_LABEL[s]} | {v['n']} | {v['est']:,.0f} | {fmt_pct(s1['est'], s1['lo'], s1['hi'])} | {fmt_pct(s3['est'], s3['lo'], s3['hi'])} | {fmt_pct(s7['est'], s7['lo'], s7['hi'])} | {fmt_pct(l2['est'], l2['lo'], l2['hi'])} | {fmt_pct(dis['est'], dis['lo'], dis['hi'])} (n={dis['n']}) |")
    md.append("\n| V7 quintile | n | late>=2x | median V28 |\n|---|---|---|---|")
    for r in R["late_by_V7q"]:
        md.append(f"| {r['group']} | {r['n']} | {fmt_pct(r['est'], r['lo'], r['hi'])} | {R['late_by_V7q_medV28'][r['group']]:,.0f} |")
    md.append("\n| Disappearance by category (35d) | n | rate |\n|---|---|---|")
    for r in sorted(R["disappearance"]["by_category"], key=lambda r: -r["est"]):
        md.append(f"| {r['group']} | {r['n']} | {fmt_pct(r['est'], r['lo'], r['hi'])} |")
    md.append("\n| Disappearance by last-observed views | n | rate |\n|---|---|---|")
    for r in R["disappearance"]["by_vbin"]:
        md.append(f"| {r['group']} | {r['n']} | {fmt_pct(r['est'], r['lo'], r['hi'])} |")
    md.append("\n| Like-rate ratio late/early by V28 quintile | n | median ratio |\n|---|---|---|")
    for r in R["like_dilution"]["ratio_by_V28q"]:
        md.append(f"| {r['group']} | {r['n']} | {fmt_num(r['est'], r['lo'], r['hi'], 3)} |")
    md.append("\n| Increment model: log(V28-Vt) ~ log Vt | n | R2 [CI] | slope [CI] | ratio V28/Vt median / p90 |\n|---|---|---|---|---|")
    for k, p in R["increment_model"].items():
        md.append(f"| {k} | {p['n']} | {fmt_num(p['R2'], p['R2_lo'], p['R2_hi'], 3)} | {fmt_num(p['slope'], p['slope_lo'], p['slope_hi'], 3)} | {p['ratio_q'][0.5]:.2f} / {p['ratio_q'][0.9]:.2f} |")
    s_ = R["interp_sensitivity"]
    md.append(f"\nInterpolation sensitivity (median share): S1 loglinear {100*s_['S1_loglinear']:.1f} / linear {100*s_['S1_linear']:.1f} / next-obs {100*s_['S1_next_obs']:.1f} (first<=6h: {100*s_['S1_loglinear_first_le_6h']:.1f}, n={s_['n_first_le_6h']}); S3 loglinear {100*s_['S3_loglinear']:.1f} / linear {100*s_['S3_linear']:.1f} / next-obs {100*s_['S3_next_obs']:.1f}; S7 {100*s_['S7_loglinear']:.1f} / {100*s_['S7_linear']:.1f}")
    md.append("\n| Subscriber quintile | n | share d3 | share d7 | late>=2x | median V28 |\n|---|---|---|---|---|---|")
    for r3, r7, l2 in zip(R["sV3_by_subq"], R["sV7_by_subq"], R["late2_by_subq"]):
        md.append(f"| {r3['group']} | {r3['n']} | {fmt_pct(r3['est'], r3['lo'], r3['hi'])} | {fmt_pct(r7['est'], r7['lo'], r7['hi'])} | {fmt_pct(l2['est'], l2['lo'], l2['hi'])} | {R['medV28_by_subq'][r3['group']]:,.0f} |")
    md.append("\n| Segment | raw share d3 | category-standardized | size-standardized |\n|---|---|---|---|")
    for s in SEG_ORDER:
        raw = next(r for r in R["seg_table"]["sV3"] if r["group"] == s); c = R["sV3_seg_category_standardized"][s]; z = R["sV3_seg_size_standardized"][s]
        md.append(f"| {SEG_LABEL[s]} | {fmt_pct(raw['est'], raw['lo'], raw['hi'])} | {fmt_pct(c['est'], c['lo'], c['hi'])} | {fmt_pct(z['est'], z['lo'], z['hi'])} |")
    md.append("\n| Late-growth threshold | share |\n|---|---|")
    for th, r in R["late_thresholds"].items():
        md.append(f"| {th} | {fmt_pct(r['est'], r['lo'], r['hi'])} |")
    km_o = R["km_overall"]
    md.append(f"\nKM cumulative disappearance (seeded<=24h, n={R['km_n']['all']}): d7 {100*km_o[7]:.1f}%, d14 {100*km_o[14]:.1f}%, d28 {100*km_o[28]:.1f}%, d35 {100*km_o[35]:.1f}%")
    md.append("by segment d35: " + ", ".join(f"{SEG_LABEL[s]} {100*R['km_by_segment'][s][35]:.1f}%" for s in SEG_ORDER))
    sv = R["survivorship"]; md.append(f"\nSurvivorship: of {sv['eligible_28d']} eligible seeds, {sv['lost_before_28d']} ({100*sv['share_lost']:.1f}%) disappeared before day 28; median V3 lost {sv['medV3_lost']:,.0f} vs cohort {sv['medV3_cohort']:,.0f}")
    mc = R["music_channel_view"]; md.append(f"Music channel view: {mc['videos']} videos / {mc['channels']} channels; channels with any missing {100*mc['share_channels_with_any_missing']:.1f}%; capped(3/channel) rate {100*mc['capped3_rate']:.1f}%; top channel share of missing {100*mc['top_channel_share_of_missing']:.1f}%; top5 {100*mc['top5_channels_share_of_missing']:.1f}%")
    hl = R["hidden_likes"]; md.append(f"Hidden likes: {100*hl['share_videos_hidden']:.1f}% of videos; medV28 hidden {hl['medV28_hidden']:,.0f} vs shown {hl['medV28_shown']:,.0f}; sV3 {hl['sV3_hidden']:.3f} vs {hl['sV3_shown']:.3f}; by segment {hl['by_segment']}")
    Path(f"{stem}_tables.md").write_text("\n".join(md), encoding="utf-8")

    # ---- figures
    log("figures ...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 150, "savefig.dpi": 300, "savefig.facecolor": "white"})

    def save(fig, name):
        fig.savefig(args.figdir / f"{name}.png", bbox_inches="tight"); fig.savefig(args.figdir / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)

    # Fig 1: normalized cumulative view curves, overall + small multiples by category
    cats = sorted(keep_cats, key=lambda c: -R["curve_by_category"][c][3])
    ncol = 3; nrow = int(np.ceil((len(cats) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.5, 2.4 * nrow), sharex=True, sharey=True)
    axes = axes.ravel()
    ages = AGES
    ov = [R["curve_overall"][a]["q50"] for a in ages]
    for i, c in enumerate(["All categories"] + cats):
        ax = axes[i]
        ax.grid(True, axis="y")
        ax.plot(ages, ov, color=MUTED, lw=1.2, ls="--", label="All (median)")
        if c == "All categories":
            lo = [R["curve_overall"][a]["q25"] for a in ages]; hi = [R["curve_overall"][a]["q75"] for a in ages]
            ax.fill_between(ages, lo, hi, color=PAL[0], alpha=.18, lw=0, label="IQR")
            ax.plot(ages, ov, color=PAL[0], lw=2, label="Median")
        else:
            yy = [R["curve_by_category"][c][a] for a in ages]
            ax.plot(yy and ages, yy, color=PAL[0], lw=2)
            ax.annotate(f"d3 = {100*yy[2]:.0f}%", xy=(3, yy[2]), xytext=(9, yy[2] - .12), fontsize=8, color=INK2)
        ax.set_title(c, fontsize=9, color=INK, loc="left")
        ax.set_xticks([1, 3, 7, 14, 21, 28]); ax.set_ylim(0, 1.02); ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    for j in range(len(cats) + 1, len(axes)): axes[j].axis("off")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.supxlabel("Days since publication", color=INK2, fontsize=9); fig.supylabel("Share of day-28 views reached", color=INK2, fontsize=9)
    fig.tight_layout(); save(fig, "Fig1_cumulative_share")

    # Fig 2: A) log V3 vs log V28 (nested)  B) log V3 vs log(V28 - V3) (increment)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.6, 4.0))
    x = np.log10(traj.V3.clip(lower=1)); y = traj.logV28
    hb = ax.hexbin(x, y, gridsize=55, cmap="Blues", mincnt=1, linewidths=0.1)
    p3 = R["predictability"]["V3"]
    xx = np.linspace(x.min(), x.max(), 50); ax.plot(xx, p3["intercept"] + p3["slope"] * xx, color=PAL[1], lw=2, label=f"Fit (slope {p3['slope']:.2f}, R² {p3['R2']:.3f})")
    ax.plot(xx, xx, color=MUTED, lw=1, ls="--", label="y = x")
    ax.set_xlabel("log10 views at day 3"); ax.set_ylabel("log10 views at day 28"); ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.set_title("A", loc="left", fontweight="bold")
    gain = (traj.V28 - traj.V3).clip(lower=0); m = gain > 0
    x2 = np.log10(traj.V3[m].clip(lower=1)); y2 = np.log10(gain[m] + 1)
    hb2 = ax2.hexbin(x2, y2, gridsize=55, cmap="Blues", mincnt=1, linewidths=0.1)
    pi = R["increment_model"]["V3"]
    xx2 = np.linspace(x2.min(), x2.max(), 50); ax2.plot(xx2, pi["intercept"] + pi["slope"] * xx2, color=PAL[1], lw=2, label=f"Fit (slope {pi['slope']:.2f}, R² {pi['R2']:.3f})")
    ax2.set_xlabel("log10 views at day 3"); ax2.set_ylabel("log10 views gained from day 3 to day 28"); ax2.legend(frameon=False, fontsize=8, loc="upper left"); ax2.set_title("B", loc="left", fontweight="bold")
    cb = fig.colorbar(hb2, ax=[ax, ax2], shrink=.8, pad=0.02); cb.set_label("Videos per cell", color=INK2); cb.outline.set_visible(False)
    save(fig, "Fig2_predictability")

    # Fig 6: cumulative disappearance (Kaplan-Meier) by creator location
    fig, ax = plt.subplots(figsize=(5.4, 3.6)); ax.grid(True, axis="y")
    for i, s_ in enumerate(SEG_ORDER):
        ax.plot(R["km_grid"], [100 * v for v in R["km_by_segment"][s_]], color=PAL[i], lw=2, label=f"{SEG_LABEL[s_]} (n={R['km_n'][s_]:,})")
    ax.plot(R["km_grid"], [100 * v for v in R["km_overall"]], color=INK, lw=1.2, ls="--", label=f"All (n={R['km_n']['all']:,})")
    ax.set_xlabel("Days since publication"); ax.set_ylabel("Cumulative share no longer retrievable (%)"); ax.set_xticks([0, 7, 14, 21, 28, 35])
    ax.legend(frameon=False, fontsize=7.5, loc="upper left"); fig.tight_layout(); save(fig, "Fig6_disappearance_curve")

    # Fig 3: late growth share by V7 quintile (bars) and by category (dots)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.4), gridspec_kw={"width_ratios": [1, 1.3]})
    t = pd.DataFrame(R["late_by_V7q"])
    a1.grid(True, axis="y"); a1.bar(range(len(t)), 100 * t.est, color=PAL[0], width=.62)
    a1.errorbar(range(len(t)), 100 * t.est, yerr=[100 * (t.est - t.lo), 100 * (t.hi - t.est)], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    a1.set_xticks(range(len(t))); a1.set_xticklabels([s.replace(" (lowest)", "\n(lowest)").replace(" (highest)", "\n(highest)") for s in t.group], fontsize=8)
    a1.set_ylabel("Videos with V28 / V7 >= 2 (%)"); a1.set_xlabel("Quintile of views at day 7"); a1.set_title("A", loc="left", fontweight="bold")
    t2 = pd.DataFrame(R["late_by_category"]).sort_values("est")
    a2.grid(True, axis="x"); a2.hlines(range(len(t2)), 100 * t2.lo, 100 * t2.hi, color=INK2, lw=1); a2.plot(100 * t2.est, range(len(t2)), "o", color=PAL[0], ms=6)
    a2.set_yticks(range(len(t2))); a2.set_yticklabels(t2.group, fontsize=8); a2.set_xlabel("Videos with V28 / V7 >= 2 (%)"); a2.set_title("B", loc="left", fontweight="bold")
    fig.tight_layout(); save(fig, "Fig3_late_growth")

    # Fig 4: disappearance by segment and category
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.4), gridspec_kw={"width_ratios": [1, 1.3]})
    t = pd.DataFrame(R["disappearance"]["by_segment"]); t["label"] = t.group.map(SEG_LABEL)
    a1.grid(True, axis="x"); a1.hlines(range(len(t)), 100 * t.lo, 100 * t.hi, color=INK2, lw=1); a1.plot(100 * t.est, range(len(t)), "o", color=PAL[1], ms=6)
    a1.set_yticks(range(len(t))); a1.set_yticklabels(t.label, fontsize=8); a1.set_xlabel("Disappeared within 35 days (%)"); a1.set_title("A  Creator location", loc="left", fontweight="bold"); a1.invert_yaxis()
    t2 = pd.DataFrame(R["disappearance"]["by_category"]).sort_values("est")
    a2.grid(True, axis="x"); a2.hlines(range(len(t2)), 100 * t2.lo, 100 * t2.hi, color=INK2, lw=1); a2.plot(100 * t2.est, range(len(t2)), "o", color=PAL[1], ms=6)
    a2.set_yticks(range(len(t2))); a2.set_yticklabels(t2.group, fontsize=8); a2.set_xlabel("Disappeared within 35 days (%)"); a2.set_title("B  Category", loc="left", fontweight="bold")
    fig.tight_layout(); save(fig, "Fig4_disappearance")

    # Fig 5: like rate by age for reach quintiles
    fig, ax = plt.subplots(figsize=(5.2, 3.6)); ax.grid(True, axis="y")
    for i, c in enumerate(lr_curve.columns):
        s = lr_curve[c].dropna(); ax.plot([float(x) for x in s.index.astype(str)], 100 * s.values, color=PAL[i], lw=2, marker="o", ms=4, label=str(c))
        ax.annotate(str(c).split(" ")[0], xy=(float(str(s.index[-1])), 100 * s.values[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=INK2, va="center")
    ax.set_xlabel("Days since publication"); ax.set_ylabel("Median likes per 100 views"); ax.set_xticks([1, 3, 7, 14, 21, 28, 35])
    ax.legend(title="Day-28 view quintile", frameon=False, fontsize=7.5, title_fontsize=8, ncol=2)
    fig.tight_layout(); save(fig, "Fig5_like_rate_dilution")

    # S1 Fig: observations per run date (design)
    fig, ax = plt.subplots(figsize=(7, 2.4)); ax.grid(True, axis="y")
    rd = runs.groupby("date").n.sum()
    ax.bar(pd.to_datetime(rd.index), rd.values, color=PAL[0], width=0.8)
    ax.set_ylabel("Observations per day"); ax.set_xlabel("Observation date (2026)")
    fig.autofmt_xdate(); fig.tight_layout(); save(fig, "S1Fig_observation_schedule")
    log("done")
    print(json.dumps({"stem": str(stem), "N": S["N"], "channels": S["channels"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
