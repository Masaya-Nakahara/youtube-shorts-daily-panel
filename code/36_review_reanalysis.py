#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""36_review_reanalysis.py -- Re-analyses requested by the pre-submission negative review (2026-10-04).

A. Disappearance v2: panel-based 24 h cohort; event time = first failed retrieval; videos aged out of the window
   with an unconfirmed missing streak are probed on the watch page and counted when confirmed unavailable;
   Kaplan–Meier with channel-bootstrap CI; all Table 8 breakdowns recomputed.
B. Kaplan–Meier by seed month at common, observable time points (September has no 35-day support).
C. Increment model: n per model, with/without zero increments, with/without +1.
D. Hidden likes: check the raw API responses for the presence of statistics.likeCount.
E. Like-rate curve with one value per video per target age (nearest observation), n and CI; per-video early/late rates.
F. Monotonisation sensitivity: trajectories without the running maximum; videos with any decrease excluded.
G. Joint standardisation (category x subscriber quintile) of the day-3 share by creator location.
H. Survivorship: composition of videos lost before day 28 vs the analytic sample.
I. Cohort definitions: ledger seed age vs panel first-observation age.

Usage: python scripts/36_review_reanalysis.py [--boot 1000] [--km-boot 200] [--probe]
Output: reports/review_reanalysis_<ts>.json, data/probe/unconfirmed_<ts>.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


HERE = Path(__file__).resolve().parent
m28 = load_mod("m28", HERE / "28_panel_lifecycle.py")
m30 = load_mod("m30", HERE / "30_probe_missing_reason.py")
STEM = "panel_lifecycle_20261003_231558"
AGES = m28.AGES; SEG_ORDER = m28.SEG_ORDER


def log(*a):
    print(datetime.now().strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def cboot(values, clusters, stat, B, rng):
    """Channel-cluster bootstrap of stat(values) -> (est, lo, hi)."""
    values = np.asarray(values, float); clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    idx_by_c = [[] for _ in uniq]
    for i, c in enumerate(inv): idx_by_c[c].append(i)
    idx_by_c = [np.array(x) for x in idx_by_c]
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx_by_c[p] for p in pick])
        out.append(stat(values[sel]))
    return float(stat(values)), float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def rate_table(d, by, col, clcol, B, rng, order=None, min_n=1):
    rows = []
    groups = [(None, d)] if by is None else list(d.groupby(by, observed=True))
    for g, dd in groups:
        if len(dd) < min_n: continue
        e, lo, hi = cboot(dd[col].to_numpy(float), dd[clcol].to_numpy(), np.nanmean, B, rng)
        rows.append({"group": "all" if g is None else str(g), "n": int(len(dd)), "n_channels": int(dd[clcol].nunique()), "est": e, "lo": lo, "hi": hi})
    t = pd.DataFrame(rows)
    if order is not None and by is not None:
        t["_o"] = t.group.map({g: i for i, g in enumerate(order)}); t = t.sort_values("_o").drop(columns="_o")
    return t.to_dict("records")


def km(times, events, grid):
    times = np.asarray(times, float); events = np.asarray(events, int)
    ev_t = np.unique(times[events == 1]); S = 1.0; keys, vals = [], []
    for ut in ev_t:
        at_risk = int(np.sum(times >= ut)); d = int(np.sum((times == ut) & (events == 1)))
        if at_risk > 0: S *= (1 - d / at_risk)
        keys.append(ut); vals.append(S)
    keys = np.array(keys); vals = np.array(vals); out = []
    for g in grid:
        k = np.searchsorted(keys, g, side="right"); out.append(0.0 if k == 0 else 1 - float(vals[k - 1]))
    return out


def km_boot(d, grid, points, B, rng):
    """Channel-bootstrap CI for KM at selected points."""
    uniq, inv = np.unique(d.channel_id.to_numpy(), return_inverse=True)
    idx_by_c = [[] for _ in uniq]
    for i, c in enumerate(inv): idx_by_c[c].append(i)
    idx_by_c = [np.array(x) for x in idx_by_c]
    t = d.time.to_numpy(float); e = d.event.to_numpy(int)
    base = km(t, e, points)
    reps = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq)); sel = np.concatenate([idx_by_c[p] for p in pick])
        reps.append(km(t[sel], e[sel], points))
    reps = np.array(reps)
    return {str(p): {"est": base[i], "lo": float(np.percentile(reps[:, i], 2.5)), "hi": float(np.percentile(reps[:, i], 97.5))} for i, p in enumerate(points)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--km-boot", type=int, default=200)
    ap.add_argument("--probe", action="store_true", help="probe watch pages of videos aged out with an unconfirmed missing streak")
    ap.add_argument("--probe-delay", type=float, default=1.0)
    args = ap.parse_args()
    rng = np.random.default_rng(0); B = args.boot
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT = {"generated_at": ts, "boot": B}

    log("loading ...")
    df = m28.load_panel(ROOT / "data/panel/panel_observations.jsonl")
    df["observed_at"] = pd.to_datetime(df.observed_at, utc=True); df["published_at"] = pd.to_datetime(df.published_at, utc=True)
    led = pd.DataFrame.from_dict(json.load(open(ROOT / "state/panel_cohort.json", encoding="utf-8"))["videos"], orient="index"); led.index.name = "video_id"
    led["published_at"] = pd.to_datetime(led.published_at, utc=True)
    meta = m28.load_meta(str(ROOT / "data/raw/daily_2026*.jsonl"))
    if "video_id" in meta.columns: meta = meta.set_index("video_id")
    traj = pd.read_parquet(ROOT / "reports" / f"{STEM}_traj.parquet"); traj.index.name = "video_id"
    R = json.load(open(ROOT / "reports" / f"{STEM}.json", encoding="utf-8"))
    end = df.observed_at.max()
    run_times = np.sort(df.groupby("run_id").observed_at.first().to_numpy())

    # ---------------------------------------------------------------- per-video panel summary
    g = df.sort_values(["video_id", "observed_at"]).groupby("video_id")
    per = g.agg(first_age=("age_days", "min"), last_age=("age_days", "max"), last_obs=("observed_at", "max"), n_obs=("views", "size"),
                last_views=("views", "last"), pub=("published_at", "first"), channel_id=("channel_id", "first"),
                category=("category_name", "first"), dur=("duration_s", "first"))
    per = per.join(led[["seed_age_days", "retired", "retired_reason", "missing_streak"]], how="left")
    per = per.join(meta[["ch_country"]], how="left"); per["seg"] = m28.segment(per.ch_country)
    per["elapsed_end"] = (end - per.pub).dt.total_seconds() / 86400
    # first failed retrieval = first run after the last successful observation
    nxt_idx = np.searchsorted(run_times, per.last_obs.to_numpy(), side="right")
    nxt = np.where(nxt_idx < len(run_times), run_times[np.minimum(nxt_idx, len(run_times) - 1)], np.datetime64("NaT"))
    per["first_miss_age"] = (pd.to_datetime(nxt, utc=True) - per.pub).dt.total_seconds() / 86400

    # I. cohort definitions
    early_panel = per[per.first_age <= 1.0]; early_ledger = per[per.seed_age_days <= 1.0]
    OUT["cohort_definitions"] = {"panel_first_age_le_1d": int(len(early_panel)), "ledger_seed_age_le_1d": int(len(early_ledger)),
                                 "both": int(len(early_panel.index.intersection(early_ledger.index))),
                                 "note": "seed_age_days is measured at the collection run that found the video; first_age at the first panel measurement, which can be a few hours later"}

    # ---------------------------------------------------------------- A. disappearance v2
    log("disappearance v2 ...")
    unconf = per[(per.retired_reason == "aged_out") & (per.missing_streak.fillna(0) > 0)]
    probe_path = ROOT / "data/probe" / f"unconfirmed_{ts}.jsonl"
    prev = sorted((ROOT / "data/probe").glob("unconfirmed_*.jsonl"))
    probe_res = {}
    if prev:
        for line in open(prev[-1], encoding="utf-8"):
            r = json.loads(line); probe_res[r["video_id"]] = r
    todo = [v for v in unconf.index if v not in probe_res]
    if args.probe and todo:
        log(f"probing {len(todo)} videos aged out with an unconfirmed missing streak ...")
        s = requests.Session(); s.headers.update({"User-Agent": m30.UA, "Accept-Language": "en-US,en;q=0.9,ja;q=0.8"})
        s.cookies.set("CONSENT", "YES+cb", domain=".youtube.com"); s.cookies.set("SOCS", "CAI", domain=".youtube.com")
        with open(probe_path, "a", encoding="utf-8") as f:
            for vid in todo:
                rec = m30.probe(s, vid); rec["fetched_at"] = datetime.utcnow().isoformat(); probe_res[vid] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); time.sleep(args.probe_delay)
    unconf_gone = {v for v, r in probe_res.items() if r.get("http") == 200 and not r.get("has_video_details", True)}
    unconf_exists = {v for v, r in probe_res.items() if r.get("has_video_details")}
    OUT["unconfirmed_at_window_end"] = {"n": int(len(unconf)), "n_in_24h_cohort": int(len(unconf.index.intersection(early_panel.index))),
                                        "probed": int(len([v for v in unconf.index if v in probe_res])), "confirmed_unavailable": int(len(unconf_gone & set(unconf.index))),
                                        "exists_at_probe": int(len(unconf_exists & set(unconf.index))),
                                        "streak_counts": unconf.missing_streak.value_counts().to_dict()}

    E = early_panel.copy()
    E["event_confirmed"] = (E.retired_reason == "missing")
    E["event_probe"] = E.index.isin(unconf_gone)
    E["event"] = (E.event_confirmed | E.event_probe).astype(int)
    E["time"] = np.where(E.event == 1, E.first_miss_age, np.minimum(E.elapsed_end, 35.0))
    E = E[E.time.notna()]
    done35 = E[E.elapsed_end >= 35]; done7 = E[E.elapsed_end >= 7]
    done35 = done35.assign(miss=done35.event.astype(float), miss_confirmed=done35.event_confirmed.astype(float),
                           miss_by7=((done35.event == 1) & (done35.first_miss_age <= 7)).astype(float))
    done7 = done7.assign(miss7=((done7.event == 1) & (done7.first_miss_age <= 7)).astype(float))
    A = {"n_35d": int(len(done35)), "events_35d": int(done35.miss.sum()), "events_confirmed_3miss": int(done35.miss_confirmed.sum()),
         "events_probe_confirmed": int((done35.event_probe).sum()),
         "rate_35d": rate_table(done35, None, "miss", "channel_id", B, rng)[0],
         "rate_35d_confirmed_only": rate_table(done35, None, "miss_confirmed", "channel_id", B, rng)[0],
         "rate_7d": rate_table(done7, None, "miss7", "channel_id", B, rng)[0], "n_7d": int(len(done7)),
         "by_segment": rate_table(done35, "seg", "miss", "channel_id", B, rng, order=SEG_ORDER),
         "by_category": rate_table(done35, "category", "miss", "channel_id", B, rng, min_n=300),
         "by_country_top": sorted(rate_table(done35.assign(cc=done35.ch_country.fillna("NA")), "cc", "miss", "channel_id", B, rng, min_n=300), key=lambda r: -r["est"]),
         "first_miss_minus_last_success_days_q": (done35[done35.event == 1].first_miss_age - done35[done35.event == 1].last_age).quantile([.1, .5, .9]).round(3).to_dict(),
         "missing_first_miss_age_q": done35[done35.event == 1].first_miss_age.quantile([.1, .25, .5, .75, .9]).round(2).to_dict(),
         "missing_last_views_q": done35[done35.event == 1].last_views.quantile([.1, .5, .9]).to_dict(),
         "missing_last_views_sum": float(done35[done35.event == 1].last_views.sum())}
    vb = pd.cut(done35.last_views, [-1, 1e3, 1e4, 1e5, 1e6, np.inf], labels=["<1k", "1k-10k", "10k-100k", "100k-1M", ">=1M"])
    A["by_last_views"] = rate_table(done35.assign(vb=vb), "vb", "miss", "channel_id", B, rng, order=["<1k", "1k-10k", "10k-100k", "100k-1M", ">=1M"])
    db = pd.cut(done35.dur, [0, 15, 30, 45, 60, np.inf], labels=["<=15 s", "16-30 s", "31-45 s", "46-60 s", ">60 s"])
    A["by_duration"] = rate_table(done35.assign(db=db), "db", "miss", "channel_id", B, rng, order=["<=15 s", "16-30 s", "31-45 s", "46-60 s"])
    # channel concentration
    capped = done35.sample(frac=1, random_state=0).groupby("channel_id").head(3)  # at most 3 videos per channel
    top5 = done35[done35.event == 1].groupby("channel_id").size().sort_values(ascending=False).head(5)
    A["rate_capped_3_per_channel"] = float(capped.miss.mean()); A["rate_excl_top5_channels"] = float(done35[~done35.channel_id.isin(top5.index)].miss.mean())
    A["top5_channel_events"] = top5.to_dict()
    mus = done35[done35.category == "Music"]; topm = mus[mus.event == 1].groupby("channel_id").size().sort_values(ascending=False)
    A["music"] = {"n": int(len(mus)), "rate": float(mus.miss.mean()), "top_channel_events": int(topm.iloc[0]) if len(topm) else 0,
                  "top_channel_n": int((mus.channel_id == topm.index[0]).sum()) if len(topm) else 0,
                  "rate_excl_top_channel": float(mus[mus.channel_id != topm.index[0]].miss.mean()) if len(topm) else None,
                  "rate_capped": float(mus.sample(frac=1, random_state=0).groupby("channel_id").head(3).miss.mean()),
                  "rate_channel_unit": float(mus.groupby("channel_id").miss.mean().mean())}
    if len(topm):
        A["by_segment_excl_music_top_channel"] = rate_table(done35[done35.channel_id != topm.index[0]], "seg", "miss", "channel_id", B, rng, order=SEG_ORDER)
    A["by_segment_capped_3_per_channel"] = rate_table(capped, "seg", "miss", "channel_id", min(B, 300), rng, order=SEG_ORDER)
    # KM with first-miss event time
    grid = list(range(0, 36)); pts = [7, 14, 28, 33, 35]
    A["km_grid"] = grid; A["km_overall"] = km(E.time, E.event, grid)
    A["km_by_segment"] = {s: km(E[E.seg == s].time, E[E.seg == s].event, grid) for s in SEG_ORDER}
    A["km_n"] = {"all": int(len(E)), **{s: int((E.seg == s).sum()) for s in SEG_ORDER}, "events": int(E.event.sum())}
    A["km_at_risk"] = {str(p): int((E.time >= p).sum()) for p in pts}
    A["km_ci"] = km_boot(E, grid, pts, args.km_boot, rng)
    A["km_ci_by_segment"] = {s: km_boot(E[E.seg == s], grid, [7, 28, 35], args.km_boot, rng) for s in SEG_ORDER}
    OUT["disappearance_v2"] = A
    E[["first_age", "last_age", "first_miss_age", "elapsed_end", "event", "event_confirmed", "event_probe", "time", "seg", "category", "channel_id", "last_views"]].to_parquet(ROOT / "reports" / f"review_disappearance_{ts}.parquet")
    A["per_video_file"] = f"reports/review_disappearance_{ts}.parquet"

    # ---------------------------------------------------------------- B. seed-month KM at observable points
    E["seed_month"] = E.pub.dt.tz_convert("Asia/Tokyo").dt.strftime("%Y-%m")
    Bm = {}
    for mth, d in E.groupby("seed_month"):
        maxage = float(d.time.max()); pts_m = [p for p in [7, 14, 28, 35] if p <= maxage + 1e-9]
        kmv = km(d.time, d.event, pts_m) if pts_m else []
        d35 = d[d.elapsed_end >= 35]
        Bm[mth] = {"n": int(len(d)), "max_observed_age": round(maxage, 2), "km": {str(p): v for p, v in zip(pts_m, kmv)},
                   "n_passed_35d": int(len(d35)), "crude_rate_35d": float(d35.event.mean()) if len(d35) else None,
                   "not_estimable": [p for p in [7, 14, 28, 35] if p > maxage]}
    OUT["seed_month_v2"] = Bm

    # ---------------------------------------------------------------- C. increment model variants
    log("increment model variants ...")
    C = {}
    for k in ("V1", "V2", "V3", "V7"):
        gain = (traj.V28 - traj[k]).clip(lower=0).to_numpy(); x = np.log10(traj[k].clip(lower=1).to_numpy())
        pos = gain > 0
        def r2(xv, yv):
            b, a = np.polyfit(xv, yv, 1); yh = a + b * xv; return float(1 - np.sum((yv - yh) ** 2) / np.sum((yv - yv.mean()) ** 2)), float(b)
        r_pos1, b_pos1 = r2(x[pos], np.log10(gain[pos] + 1)); r_pos0, b_pos0 = r2(x[pos], np.log10(gain[pos])); r_all1, b_all1 = r2(x, np.log10(gain + 1))
        C[k] = {"n_total": int(len(gain)), "n_positive": int(pos.sum()), "n_zero": int((~pos).sum()),
                "R2_pos_log1p": r_pos1, "slope_pos_log1p": b_pos1, "R2_pos_log": r_pos0, "slope_pos_log": b_pos0, "R2_all_log1p": r_all1, "slope_all_log1p": b_all1}
    OUT["increment_model_variants"] = C

    # ---------------------------------------------------------------- D. hidden likes vs raw API responses
    log("hidden likes check (raw files) ...")
    import glob
    want = set(traj.index); has_like = {}
    for f in sorted(glob.glob(str(ROOT / "data/raw/daily_2026*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            try: d = json.loads(line)
            except Exception: continue
            vid = d.get("id")
            if vid in want and vid not in has_like:
                st = d.get("statistics") or {}
                has_like[vid] = "likeCount" in st
    hl = pd.Series(has_like).rename("likecount_present")
    chk = traj[["likes_hidden"]].join(hl, how="left")
    OUT["hidden_likes_check"] = {"n_matched_raw": int(chk.likecount_present.notna().sum()),
                                 "crosstab": {f"hidden={h}": chk[chk.likes_hidden == h].likecount_present.value_counts(dropna=False).rename(lambda x: str(x)).to_dict() for h in (True, False)},
                                 "share_flagged_without_likecount": float((chk[chk.likes_hidden == True].likecount_present == False).mean()),
                                 "share_unflagged_with_likecount": float((chk[chk.likes_hidden == False].likecount_present == True).mean())}

    # ---------------------------------------------------------------- E. like-rate curve, one value per video per age
    log("like-rate v2 ...")
    dd = df[df.video_id.isin(traj.index)].copy()
    dd["lr"] = np.where((dd.views > 1000) & (dd.likes > 0), dd.likes / dd.views, np.nan)
    targets = [1, 2, 3, 5, 7, 10, 14, 21, 28, 35]
    recs = []
    for t in targets:
        w = dd[(dd.age_days - t).abs() <= 0.5].copy(); w["dist"] = (w.age_days - t).abs()
        w = w.sort_values("dist").drop_duplicates("video_id")
        w = w[w.lr.notna()][["video_id", "lr"]].assign(t=t); recs.append(w)
    L = pd.concat(recs)
    L = L.join(traj[["V28q", "channel"]], on="video_id")
    curve = {}
    for q, dq in L.groupby("V28q", observed=True):
        curve[str(q)] = {}
        for t, dt in dq.groupby("t"):
            e, lo, hi = cboot(dt.lr.to_numpy(), dt.channel.to_numpy(), np.nanmedian, min(B, 300), rng)
            curve[str(q)][str(t)] = {"n": int(len(dt)), "median": e, "lo": lo, "hi": hi}
    # balanced subset: videos with a value at 1, 3, 7, 14, 28
    piv = L.pivot(index="video_id", columns="t", values="lr")
    bal = piv.dropna(subset=[1, 3, 7, 14, 28])
    balq = traj.V28q.reindex(bal.index)
    OUT["like_rate_v2"] = {"curve": curve, "balanced_n": int(len(bal)),
                           "balanced_curve": {str(q): {str(t): float(bal.loc[balq == q, t].median()) for t in [1, 3, 7, 14, 28]} for q in balq.dropna().unique()},
                           "balanced_ratio_28_over_1_median": float((bal[28] / bal[1]).median()),
                           "balanced_share_declined": float((bal[28] < bal[1]).mean())}
    # per-video early/late (paper definition) for the public data
    d2 = df[(df.views > 1000) & (df.likes > 0)].copy(); d2["lr"] = d2.likes / d2.views
    e1 = d2[d2.age_days <= 1.5].sort_values("age_days").groupby("video_id").agg(lr_early=("lr", "first"), age_early=("age_days", "first"))
    l1 = d2[d2.age_days >= 25].sort_values("age_days").groupby("video_id").agg(lr_late=("lr", "last"), age_late=("age_days", "last"))
    pv = traj[[]].join(e1, how="left").join(l1, how="left")
    pv["like_ratio_late_over_early"] = pv.lr_late / pv.lr_early
    pv = pv.join(piv.rename(columns=lambda t: f"lr_day{t}"), how="left")
    pv.to_parquet(ROOT / "reports" / f"review_like_rates_{ts}.parquet")
    OUT["like_rate_v2"]["per_video_file"] = f"reports/review_like_rates_{ts}.parquet"
    OUT["like_rate_v2"]["age_early_q"] = pv.age_early.quantile([.1, .5, .9]).round(2).to_dict(); OUT["like_rate_v2"]["age_late_q"] = pv.age_late.quantile([.1, .5, .9]).round(2).to_dict()

    # ---------------------------------------------------------------- F. monotonisation sensitivity
    log("monotonisation sensitivity ...")
    sub = df[df.video_id.isin(traj.index)].sort_values(["video_id", "age_days"])
    rows = []; any_dec = {}
    for vid, grp in sub.groupby("video_id", sort=False):
        a = grp.age_days.to_numpy(); v = grp.views.to_numpy(dtype=float)
        any_dec[vid] = bool((np.diff(v) < 0).any())
        inter = np.expm1(np.interp([3, 7, 28], a, np.log1p(np.maximum(v, 0)), left=np.nan, right=np.nan))
        rows.append((vid, *inter))
    raw = pd.DataFrame(rows, columns=["video_id", "V3r", "V7r", "V28r"]).set_index("video_id")
    raw["any_decrease"] = pd.Series(any_dec)
    raw = raw.join(traj[["V3", "V7", "V28", "sV3", "sV7", "late2", "channel"]])
    raw["sV3r"] = raw.V3r / raw.V28r; raw["sV7r"] = raw.V7r / raw.V28r; raw["late2r"] = (raw.V28r / raw.V7r >= 2).astype(float); raw["ratio3r"] = raw.V28r / raw.V3r.clip(lower=1)
    def summ(d):
        return {"n": int(len(d)), "sV3_median": float(d.sV3r.median()), "sV7_median": float(d.sV7r.median()), "late2_share": float(d.late2r.mean()),
                "ratio_V28_V3_median": float(d.ratio3r.median()), "ratio_V28_V3_p90": float(d.ratio3r.quantile(.9)),
                "R2_cum_V3": float(np.corrcoef(np.log10(d.V3r.clip(lower=1)), np.log10(d.V28r.clip(lower=1)))[0, 1] ** 2)}
    OUT["monotonisation_sensitivity"] = {"videos_with_any_decrease": int(raw.any_decrease.sum()), "share": float(raw.any_decrease.mean()),
                                         "monotonised_reference": {"sV3_median": float(traj.sV3.median()), "sV7_median": float(traj.sV7.median()), "late2_share": float(traj.late2.mean()),
                                                                   "ratio_V28_V3_median": float((traj.V28 / traj.V3.clip(lower=1)).median())},
                                         "raw_all": summ(raw.dropna(subset=["V3r", "V28r"])), "raw_excluding_decreases": summ(raw[~raw.any_decrease].dropna(subset=["V3r", "V28r"]))}

    # ---------------------------------------------------------------- G. joint standardisation (category x subq)
    log("joint standardisation ...")
    cell = traj.groupby(["category", "subq"], observed=True).size(); w = cell / cell.sum()
    def joint_std(d, min_cell=5):
        med = d.groupby(["category", "subq"], observed=True).sV3.agg(["median", "size"])
        med = med[med["size"] >= min_cell]
        ww = w.reindex(med.index).fillna(0)
        return float((ww * med["median"]).sum() / ww.sum()), float(ww.sum())
    G = {}
    for s in SEG_ORDER:
        d = traj[traj.seg == s]; est, cover = joint_std(d)
        reps = []
        uniq = d.channel.unique(); ch_groups = {c: idx for c, idx in d.groupby("channel").groups.items()}
        for _ in range(min(B, 300)):
            pick = rng.choice(uniq, len(uniq), replace=True)
            dd_ = d.loc[np.concatenate([ch_groups[c] for c in pick])]
            reps.append(joint_std(dd_)[0])
        G[s] = {"est": est, "lo": float(np.percentile(reps, 2.5)), "hi": float(np.percentile(reps, 97.5)), "weight_coverage": cover, "n": int(len(d))}
    OUT["joint_standardisation_sV3"] = G

    # ---------------------------------------------------------------- H. survivorship composition
    log("survivorship ...")
    elig = per[(per.first_age <= 1.0) & (per.elapsed_end >= 28)]
    lost = elig[(elig.retired_reason == "missing") & (elig.last_age < 28)]
    comp = {}
    for col in ["category", "seg"]:
        comp[col] = {"lost": lost[col].value_counts(normalize=True).round(4).to_dict(), "analytic": traj[col if col != "seg" else "seg"].value_counts(normalize=True).round(4).to_dict()}
    comp["seed_month"] = {"lost": lost.pub.dt.tz_convert("Asia/Tokyo").dt.strftime("%Y-%m").value_counts(normalize=True).round(4).to_dict(),
                          "analytic": per.loc[per.index.intersection(traj.index)].pub.dt.tz_convert("Asia/Tokyo").dt.strftime("%Y-%m").value_counts(normalize=True).round(4).to_dict()}
    comp["first_age_median"] = {"lost": float(lost.first_age.median()), "analytic": float(elig.loc[elig.index.intersection(traj.index)].first_age.median())}
    comp["lost_last_age_q"] = lost.last_age.quantile([.1, .25, .5, .75, .9]).round(2).to_dict()
    comp["counts"] = {"eligible_28d": int(len(elig)), "lost_before_28d": int(len(lost)), "lost_with_V3": R["survivorship"]["lost_with_V3"],
                      "lost_before_3d": int((lost.last_age < 3).sum())}
    OUT["survivorship_v2"] = comp

    outp = ROOT / "reports" / f"review_reanalysis_{ts}.json"
    outp.write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log("wrote", outp)
    print(json.dumps({"out": str(outp), "rate_35d": A["rate_35d"], "rate_35d_confirmed_only": A["rate_35d_confirmed_only"]["est"], "km": A["km_ci"],
                      "unconfirmed": OUT["unconfirmed_at_window_end"], "hidden_check": OUT["hidden_likes_check"]["share_flagged_without_likecount"],
                      "increment_V3": C["V3"], "mono": OUT["monotonisation_sensitivity"]["raw_all"], "joint": {k: round(v["est"], 4) for k, v in G.items()},
                      "cohorts": OUT["cohort_definitions"]}, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
