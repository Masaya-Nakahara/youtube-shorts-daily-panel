# youtube-shorts-daily-panel

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23133817.svg)](https://doi.org/10.5281/zenodo.23133817)

Code and derived data for a daily panel of #shorts-labeled YouTube videos surfaced by a fixed Japan-region search configuration, observed once a day from June to October 2026. The panel follows each video from shortly after publication for up to 35 days and records views, likes, comments and availability. The derived data describe the post-publication view dynamics and the disappearance (deletion or privatisation) of 26,705 videos first observed within 24 hours of publication. Details of the associated article will be added when it is published. Package built on 2026-10-04.

## What is here

| Path | Content |
|---|---|
| `code/01_collect_pilot.py` | Daily retrieval of new #shorts-labeled videos through the YouTube Data API v3 (`search.list`, q="#shorts", regionCode=JP, relevanceLanguage=ja, 14 categories × 3 orders) |
| `code/07_panel_resample.py` | Panel maintenance: seeds newly retrieved videos and re-observes every tracked video once a day (`videos.list`, `channels.list`); retires a video 35 days after publication or after three consecutive failed retrievals |
| `code/run_shorts_daily.bat` | The scheduled daily job (22:00 JST) that chains the two steps above |
| `code/28_panel_lifecycle.py` | Analysis: analytic sample, interpolation, cumulative shares, regressions, late growth, like-rate change, disappearance, Kaplan–Meier, channel-cluster bootstrap, figures and tables |
| `code/30_probe_missing_reason.py` | Fetches the public watch page of each disappeared video and classifies the displayed reason |
| `code/31_table9_from_probe.py` | Aggregates the probe output into a table of displayed reasons |
| `data/video_ids.csv` | All 58,541 tracked videos: ID, publication time, category, declared creator country and 5-class creator location, audio language, tracking status, elapsed days at panel end, membership in the analytic sample |
| `data/per_video_metrics.csv` | Per-video measures for the 26,705-video analytic sample: interpolated views at days 1–28 (log-linear, linear and next-observation variants), cumulative shares, power-law exponent, late-growth and re-acceleration flags, quintiles, hidden-likes flag |
| `data/disappearance_cohort.csv` | The 47,586 videos first observed within 24 h of publication: cohort flags (7 d, 35 d), disappearance status, age at last observation, watch-page probe class for disappeared videos |

Channels are identified only by an anonymised integer `channel_key`, which is sufficient to reproduce the channel-cluster bootstrap.

## What is not here, and how to get it

The daily raw observations (view, like and comment counts per video per day) were obtained under the YouTube API Services Terms of Service, which require stored data to be refreshed or deleted every 30 days. They are therefore not redistributed. The same series can be re-collected with the code in `code/`:

1. Obtain a YouTube Data API v3 key and store it in the environment variable `YOUTUBE_API_KEY` (or in `~/.config/youtube_api/key`).
2. Install dependencies: `pip install -r requirements.txt` (Python 3.11 or later).
3. Run `code/run_shorts_daily.bat` (or the two Python steps it contains) once a day. Raw files go to `data/raw/`, the panel to `data/panel/panel_observations.jsonl`, the ledger to `state/panel_cohort.json`.
4. After at least 35 days, run `python code/28_panel_lifecycle.py --boot 1000 --seed 0` to produce the full set of tables and figures for the new series. Results are written to `reports/`.

To check the IDs in `data/video_ids.csv` against the current state of YouTube, `code/30_probe_missing_reason.py` can be pointed at the ledger produced by step 3, or adapted to read the CSV.

## Reproducing the summary statistics from the derived data

The view-dynamics results (cumulative shares, early-view regressions, late growth, like-rate change) are summaries of `data/per_video_metrics.csv`: medians, shares and least-squares fits on log10 views, with channel-cluster bootstrap intervals over `channel_key`. The disappearance results are summaries of `data/disappearance_cohort.csv`, and the breakdown of displayed reasons is the tabulation of `probe_class` for rows with `disappeared == True`. The panel ended at 2026-10-03T13:01:24.597670+00:00; 27,832 videos had passed 35 days since publication, of which 1,384 had disappeared.

## Column notes

- `V1` … `V28`: views at exactly t days after publication, obtained by log-linear interpolation of the two adjacent daily observations after monotonisation (running maximum). `V*_lin`: linear interpolation. `V1_next`, `V3_next`: first observation after day 1 / day 3.
- `sV1`, `sV3`, `sV7`, `sV14`: cumulative share V_t / V_28.
- `power_law_exponent_b`: slope of log views on log elapsed days within the video.
- `ratio_V28_over_V7`, `late_growth_ge1_5x`, `late_growth_ge2x`, `late_growth_ge3x`: growth from day 7 to day 28 and threshold flags; `reaccelerated`: a daily increment after day 7 exceeded the maximum daily increment up to day 7.
- `likes_hidden`: like count was 0 in at least half of the observations with more than 1,000 views (validated against the absence of the likeCount field in the API response at collection).
- `like_rate_early`, `like_rate_early_age_days`: likes divided by views at the first observation within 1.5 days of publication with views > 1,000 and likes > 0, and its age; `like_rate_late`, `like_rate_late_age_days`: the same at the last observation on or after day 25; `like_ratio_late_over_early`: their ratio. `like_rate_day1` … `like_rate_day35`: like rate at the observation closest to each age (within 0.5 day), one value per video.
- In `disappearance_cohort.csv`: `in_ledger_24h_cohort` / `in_panel_24h_cohort`: first observed within 24 h according to the collection run / the panel record (the panel definition is used in the analysis); `disappeared`: three consecutive failed retrievals (ledger); `disappeared_v2`: three consecutive failed retrievals or unretrievable at the end of the 35-day window and confirmed unavailable on the watch page (`disappeared_confirmed_by_probe`); `first_failed_retrieval_age_days`: event time used for Kaplan–Meier; `km_time_days`: event or censoring time.
- `creator_country_declared`: ISO 3166-1 alpha-2 code declared by the channel (empty if not declared). `creator_location`: JP / US / IN / other (declared) / unknown (not declared).
- `tracking_status`: `completed_35d` (followed to 35 days), `disappeared` (three consecutive failed retrievals), `active` (still tracked at panel end).
- `probe_class` (disappeared videos only): `removed_uploader_message`, `removed_no_reason`, `account_terminated_or_closed`, `removed_policy`, `removed_copyright`, `private`, `exists_at_probe`.

## Licenses

Code: MIT License (see `LICENSE`). Data files in `data/`: CC BY 4.0 (see `data/LICENSE.txt`).

## Citation

Archived releases are available on Zenodo. The concept DOI https://doi.org/10.5281/zenodo.23133817 always resolves to the latest version; each release also has its own version DOI. Author and article details will be added here when the associated article is published.
