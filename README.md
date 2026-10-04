# youtube-shorts-daily-panel

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23133817.svg)](https://doi.org/10.5281/zenodo.23133817)

Code and derived data for a daily panel of #shorts-labeled YouTube videos surfaced by a fixed Japan-region search configuration, observed once a day from June to October 2026. The panel follows each video from shortly after publication for up to 35 days and records views, likes, comments and availability. The derived data describe the post-publication view dynamics of 26,705 videos first observed within 24 hours of publication, and the disappearance (deletion, privatisation or other unavailability) of the 47,563 videos in the 24-hour cohort. Details of the associated article will be added when it is published. Package built on 2026-10-05.

## What is here

| Path | Content |
|---|---|
| `code/01_collect_pilot.py` | Daily retrieval of new #shorts-labeled videos through the YouTube Data API v3 (`search.list`, q="#shorts", regionCode=JP, relevanceLanguage=ja, 14 categories × 3 orders; duration checked with `videos.list`, ≤ 60 s) |
| `code/07_panel_resample.py` | Panel maintenance: seeds newly retrieved videos and re-observes every tracked video once a day (`videos.list`, `channels.list`); retires a video 35 days after publication or after three consecutive failed retrievals |
| `code/run_shorts_daily.bat` | The scheduled daily job (22:00 JST) that chains the two steps above |
| `code/28_panel_lifecycle.py` | Main analysis: analytic sample, interpolation, cumulative shares, regressions, late growth, like-rate change, disappearance by the three-miss rule, Kaplan–Meier, channel-cluster bootstrap, figures and tables |
| `code/30_probe_missing_reason.py` | Fetches the public watch page of each disappeared video and classifies the displayed reason |
| `code/31_table9_from_probe.py` | Aggregates the probe output into a table of displayed reasons |
| `code/36_review_reanalysis.py` | Revised disappearance definition (event time = first failed retrieval; videos unretrievable at the end of their window checked on the watch page), Kaplan–Meier with bootstrap intervals, like-rate trajectory with one value per video per age, increment-model variants, monotonisation sensitivity, joint category-by-size standardisation, survivorship |
| `code/37_review2_reanalysis.py` | Survivorship under the revised event definition, view-weighted cumulative shares, like-rate change by stage within one group of videos, timing of disappearance events and sensitivity to the observation schedule, standardisation on common support, Kaplan–Meier bands and numbers at risk |
| `code/38_recompute_from_public_csv.py` | Recomputes the article's point estimates from the three CSV files alone (no raw observations or API access needed); `--boot N` adds channel-cluster bootstrap intervals for the headline estimates |
| `data/video_ids.csv` | All 59,250 tracked videos: ID, publication time, category, declared creator country and 5-class creator location, audio language, tracking status, elapsed days at panel end, membership in the analytic sample |
| `data/per_video_metrics.csv` | Per-video measures for the 26,705-video analytic sample: interpolated views at days 1–28 (log-linear, linear and next-observation variants), cumulative shares, power-law exponent, late-growth and re-acceleration flags, quintiles, like-count-not-retrieved flag, early and late like rates and like rates at each age |
| `data/disappearance_cohort.csv` | The 48,188 videos first observed within 24 h of publication under either of the two cohort definitions: cohort flags, disappearance status under both definitions, event and censoring times, watch-page probe class |
| `results/` | The complete numerical output of the three analysis scripts as run for the article (`results/panel_lifecycle_20261003_231558.json`, `results/review_reanalysis_20261004_194751.json`, `results/review2_reanalysis_20261004_210941.json`, and `results/recomputed_from_public_csv.json` (output of `code/38_recompute_from_public_csv.py` on the CSV files of this package)). Channel identifiers are replaced by `channel_key`. Every number in the article's tables, figures and text is in these files |

Channels are identified only by an anonymised integer `channel_key`, which is sufficient to reproduce the channel-cluster bootstrap.

## What is not here

The daily raw observations (the view, like and comment counts of each video on each day) were obtained through the YouTube Data API under the YouTube API Services Terms of Service and Developer Policies and are not redistributed. They also cannot be re-collected retrospectively, because the API returns only a video's current counts. The code in `code/` collects a new series under the same configuration:

1. Obtain a YouTube Data API v3 key and store it in the environment variable `YOUTUBE_API_KEY` (or in `~/.config/youtube_api/key`).
2. Install dependencies: `pip install -r requirements.txt` (Python 3.11 or later).
3. Run `code/run_shorts_daily.bat` (or the two Python steps it contains) once a day. Raw files go to `data/raw/`, the panel to `data/panel/panel_observations.jsonl`, the ledger to `state/panel_cohort.json`.
4. After at least 35 days, run `python code/28_panel_lifecycle.py --boot 1000 --seed 0`, which writes `reports/panel_lifecycle_<timestamp>.json` and `reports/panel_lifecycle_<timestamp>_traj.parquet`. Then run `python code/36_review_reanalysis.py --probe` and `python code/37_review2_reanalysis.py`: both read the newest `panel_lifecycle_*` outputs by default (`--stem panel_lifecycle_<timestamp>` selects another), and 37 reads the newest output of 36 (`--review` selects another). `python code/30_probe_missing_reason.py` and `python code/31_table9_from_probe.py` produce the displayed-reason table; 31 reads the newest probe file by default (`--probe` selects another). The scripts locate each other through their own directory, so `code/` can keep its name; the data folders are resolved relative to the parent of `code/`.

To check the IDs in `data/video_ids.csv` against the current state of YouTube, `code/30_probe_missing_reason.py` can be pointed at the ledger produced by step 3, or adapted to read the CSV.

## Reproducing the article's numbers from the derived data

- **View dynamics** (cumulative shares, early-view regressions, late growth, like-rate change): summaries of `data/per_video_metrics.csv`, namely medians, shares and least-squares fits on log10 views, with channel-cluster bootstrap intervals over `channel_key`. The per-video median of `sV3` is 80.5% and of `sV7` 94.3%; the view-weighted shares `sum(V3)/sum(V28)` and `sum(V7)/sum(V28)` are 46.1% and 71.0%.
- **Disappearance, definition used in the article** (`disappeared_v2`): rows with `in_panel_24h_cohort == True` form the 24-hour cohort (47,563 videos). Of these, rows with `elapsed_days_at_panel_end >= 35` form the 35-day cohort (28,413 videos), and `disappeared_v2 == True` marks the 1,507 disappearance events (1,416 by three consecutive failed retrievals, 91 unretrievable at the end of the window and confirmed unavailable on the watch page), giving 5.3%. The 7-day rate uses `elapsed_days_at_panel_end >= 7` as denominator and `first_failed_retrieval_age_days <= 7` as event. Kaplan–Meier estimates use `km_time_days` and `disappeared_v2` over the whole 24-hour cohort.
- **Displayed reasons** (Table 9 of the article): tabulate `probe_class` over rows with `disappeared_v2 == True` (2,005 rows, all classified); for the 35-day column add `elapsed_days_at_panel_end >= 35`.
- **Disappearance, three-miss rule only** (`disappeared`, cohort by the collection run `in_ledger_24h_cohort`): 28,428 videos passed 35 days, of which 1,416 disappeared (5.0%). This is the definition of version 1.0.0 of this package and is kept for comparison.

The panel ended at 2026-10-04T13:01:28.538029+00:00. Daily operation was continuous from 2026-07-14; before that date there were gaps between runs of up to 26 days (`results/review2_reanalysis_*.json`, `event_timing.run_gaps_gt_1_5d`), which is why the article reports a sensitivity analysis restricted to videos published on or after 2026-07-14.

## Checking the article's numbers from the derived data alone

`python code/38_recompute_from_public_csv.py` needs only the three CSV files and recomputes the point estimates of Tables 3–9 and S9 of the article (cumulative shares, regressions, late growth, like-rate ratios and the stage decomposition, disappearance rates, Kaplan–Meier estimates, displayed reasons, survivorship counts) into `results/recomputed_from_public_csv.json`; `--boot 1000` adds channel-cluster bootstrap intervals for the headline estimates. The copy in `results/` was produced from the CSV files of this package. Quantities that need the daily observations themselves, and therefore cannot be recomputed from this package, are the interpolation and monotonisation sensitivities (S5 Table and S8 Table, sheet b), the day-3 views of videos lost before day 28 (S8 Table, sheet f), the Music title-word split, the within-channel variance decomposition, and the observation-schedule sensitivity (S8 Table, sheet i).

## Column notes

- `V1` … `V28`: views at exactly t days after publication, obtained by log-linear interpolation of the two adjacent daily observations after monotonisation (running maximum). `V*_lin`: linear interpolation. `V1_next`, `V3_next`: first observation after day 1 / day 3.
- `sV1`, `sV3`, `sV7`, `sV14`: cumulative share V_t / V_28.
- `power_law_exponent_b`: slope of log views on log elapsed days within the video.
- `ratio_V28_over_V7`, `late_growth_ge1_5x`, `late_growth_ge2x`, `late_growth_ge3x`: growth from day 7 to day 28 and threshold flags; `reaccelerated`: a daily increment after day 7 exceeded the maximum daily increment up to day 7.
- `likes_hidden`: the recorded like count was 0 in at least half of the observations with more than 1,000 views. The collection code records 0 when the API response carries no `likeCount` field; for 98.9% of the flagged videos the first response indeed lacked the field, and 98.1% of the other videos had it. The flag therefore marks videos whose like count was not retrieved, which is consistent with, but not an independent confirmation of, the uploader having hidden the like count.
- `like_rate_early`, `like_rate_early_age_days`: likes divided by views at the first observation within 1.5 days of publication with views > 1,000 and likes > 0, and its age; `like_rate_late`, `like_rate_late_age_days`: the same at the last observation on or after day 25; `like_ratio_late_over_early`: their ratio. `like_rate_day1` … `like_rate_day35`: like rate at the observation closest to each age (within 0.5 day), one value per video.
- In `disappearance_cohort.csv`: `in_ledger_24h_cohort` / `in_panel_24h_cohort`: first observed within 24 h according to the collection run / the panel record (the panel definition is used in the article); `disappeared`: three consecutive failed retrievals (ledger); `disappeared_v2`: three consecutive failed retrievals (`disappeared_three_misses`) or unretrievable at the end of the 35-day window and confirmed unavailable on the watch page (`disappeared_confirmed_by_probe`); `first_failed_retrieval_age_days`: event time used in the article; `last_age_days`: age at the last successful retrieval (lower bound of the event time); `km_time_days`: event or censoring time.
- `creator_country_declared`: ISO 3166-1 alpha-2 code declared by the channel (empty if not declared). `creator_location`: JP / US / IN / other (declared) / unknown (not declared).
- `tracking_status`: `completed_35d` (followed to 35 days), `disappeared` (three consecutive failed retrievals), `active` (still tracked at panel end).
- `probe_class` (rows with `disappeared_v2 == True` or `disappeared == True`): `removed_uploader_message`, `removed_no_reason`, `account_terminated_or_closed`, `removed_policy`, `removed_copyright`, `private`, `exists_at_probe`. Watch pages were fetched on 2026-10-03/04; the 91 window-end videos were classified from the fetch of 2026-10-04.

## Version history

- v1.0.0 (2026-10-04): first release; disappearance by the three-miss rule.
- v1.1.0 (2026-10-04): revised disappearance definition, per-video like-rate columns, 24-hour cohort flags under both definitions.
- v1.2.0 (2026-10-04): analysis scripts 36 and 37, watch-page classes for the window-end videos, complete numerical results in `results/`, README aligned with the definitions and numbers of the article.
- v1.3.0 (2026-10-04): scripts 31, 36 and 37 resolve their inputs at run time (newest outputs by default; `--stem`, `--review`, `--probe` to choose) instead of a fixed past file name, so the steps above run in a clean environment; script 38 recomputes the article's numbers from the CSV files alone; README states which quantities need the raw observations.

## Licenses

Code: MIT License (see `LICENSE`). Data files in `data/` and `results/`: CC BY 4.0 (see `data/LICENSE.txt`).

## Citation

Archived releases are available on Zenodo. The concept DOI https://doi.org/10.5281/zenodo.23133817 always resolves to the latest version; each release also has its own version DOI. Author and article details will be added here when the associated article is published.
