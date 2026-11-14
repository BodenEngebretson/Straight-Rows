#!/usr/bin/env python3
"""Straight Rows: measurements -> flags -> intervals -> scored causes.

Install: python -m pip install pandas numpy
Explore: python detect.py --data data --output results.json
Check labels: python detect.py --data data --validate
Run artificial examples: python detect.py --self-test

Development verification is printed at runtime, never assumed to be 12/12.
These heuristic scores are NOT probabilities. Low scores and unsupported
single-symptom multipath are suppressed. Only May 6 is scored.
The default output contains diagnostic detail. --github-handle and --repo-url
also write submission.json for May 6. Confidence is a placeholder,
not a calibrated probability. All timestamps and interval boundaries use UTC.

Reading guide:
    main() reads terminal options and starts the program.
    detect() coordinates the detection steps:
        load measurements -> learn limits -> create flags -> find intervals
        -> classify candidates -> merge overlapping candidates.

Vocabulary:
    data / rows: pandas tables; each row is one tractor at one timestamp.
    flag: a True/False column describing an unusual measurement.
    interval: a start/end time range. Start is included; end is excluded.
    candidate: an interval with a predicted cause, scores, and evidence.
    event: overlapping candidates of the same cause merged together.

Pandas expressions usually operate on whole columns, not a single value.
For Boolean columns, & means AND, | means OR, and ~ means NOT.
"""
import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_START = pd.Timestamp("2026-05-06T00:00:00Z")
EVAL_END = pd.Timestamp("2026-05-07T00:00:00Z")
SECOND = pd.Timedelta(seconds=1)
ACTIONS = {
    "SPACE_WEATHER": "HALT_OR_DEGRADED_MODE",
    "JAMMING": "HALT_AND_REPORT",
    "SPOOFING": "HALT_CROSSCHECK_AND_REPORT",
    "EQUIPMENT_FAULT": "SERVICE_MACHINE_OTHERS_CONTINUE",
    "MULTIPATH": "CONTINUE_FALSE_ALARM",
}
# Sensitivity and duration settings are explicit, so they can be tuned.
DEFAULTS = {
    "minimum_cause_score": 3.0,
    "multipath_min_signal_spread": 0.45,
    "multipath_min_high_error": 0.40,
    "multipath_min_fix_degraded": 0.50,
    # A high heuristic score does not by itself mean an operator event.
    # No hard absolute-error cutoff by default: short real multipath can have
    # small deviations. This is distinct
    # from high_error, which merely exceeds a learned low noise floor.
    "multipath_min_cross_track_error_m": 0.0,
    # Legacy recurrence settings (retained for backwards-compatible config).
    # Detect a recurring common-mode signal pattern, learned only from
    # unlabelled timestamps in the pre-evaluation period.
    "multipath_period_min_seconds": 300.0,
    "multipath_period_max_seconds": 420.0,
    "multipath_period_tolerance_seconds": 8.0,
    "multipath_sync_tolerance_seconds": 4.0,
    "multipath_period_min_pairs": 30,
    "multipath_peer_tractors": 2,
    "multipath_group_gap_seconds": 420.0,
    "multipath_max_group_seconds": 3600.0,
    "iqr_multiplier": 4.0,
    "gap_seconds": 3,
    "minimum_hits": 8,
    "minimum_density": 0.35,
    "short_event_seconds": 60,
    "s4_elevated": 0.3,
    "kp_elevated": 5.0,
    "severe_iqr_multiplier": 20.0,
    "fleet_error_ratio": 4.0,
    "merge_gap_seconds": 180,
    "antenna_gap_seconds": 60,
    "antenna_minimum_hits": 30,
    "antenna_minimum_density": 0.15,
    "antenna_minimum_seconds": 120,
    "spoof_context_seconds": 300,
    "spoof_minimum_context_fraction": 0.5,
}
FLAGS = [
    "low_cn0",
    "weak_min_cn0",
    "high_error",
    "few_sats",
    "link_failure",
    "fix_degraded",
    "severe_cn0",
    "signal_spread",
    "excess_error",
]

def load_json(root, filename):
    """Support both downloaded flat files and the organizer's folder layout."""
    matches = list(root.rglob(filename))
    if len(matches) != 1:
        raise ValueError(f"Expected one {filename} below {root}, found {len(matches)}")
    return json.loads(matches[0].read_text())

def load_data(root):
    files = list(root.rglob("tractor_telemetry.csv.gz"))
    if len(files) != 1:
        raise ValueError("Expected one tractor_telemetry.csv.gz under --data.")
    # Pandas reads the compressed CSV into a table (a DataFrame).
    data = pd.read_csv(files[0])
    data["ts"] = pd.to_datetime(data["ts"], utc=True)
    if data.duplicated(["tractor_id", "ts"]).any():
        raise ValueError("Duplicate timestamps within a tractor need resolving.")
    required = ["mean_cn0_dbhz", "min_cn0_dbhz", "cross_track_error_m",
                "num_sats", "speed_mps", "base_link_ok", "fix_state"]
    missing_by_column = data[required].isna().any()
    if missing_by_column.any():
        raise ValueError("Telemetry has missing measurements. Resolve them before detection.")
    data = data.sort_values(["tractor_id", "ts"]).reset_index(drop=True)
    episodes = load_json(root, "train_episodes.json")["episodes"]
    feeds = {
        "kp": load_json(root, "sgi_indices.json")["kp"],
        "s4": load_json(root, "scintillation_series.json")["series"],
        "rfi": load_json(root, "rfi_reports.json")["reports"],
        "advisories": load_json(root, "constellation_advisories.json")["advisories"],
    }
    return data, episodes, feeds

def learn_limits(data, episodes, config):
    """Learn each tractor's normal limits before the evaluation period.

    Known disturbance episodes are excluded. Median and interquartile range
    (IQR = 75th percentile minus 25th percentile) reduce the influence of
    remaining unusual measurements.
    """
    # A mask is a True/False value for every row: True means keep that row.
    reference_mask = data["ts"] < EVAL_START
    for episode in episodes:
        affected_tractor = data["tractor_id"].isin(episode["tractors"])
        episode_start = pd.Timestamp(episode["start"])
        episode_end = pd.Timestamp(episode["end"])
        during_episode = (data["ts"] >= episode_start) & (data["ts"] < episode_end)
        reference_mask &= ~(affected_tractor & during_episode)

    limits = {}
    measurements = [
        "mean_cn0_dbhz", "min_cn0_dbhz", "cross_track_error_m", "num_sats"
    ]
    reference_rows = data.loc[reference_mask]
    for tractor, rows in reference_rows.groupby("tractor_id"):
        limits[tractor] = {}
        for measurement in measurements:
            values = rows[measurement]
            if measurement == "cross_track_error_m":
                # Left and right drift both count: use the error's magnitude.
                values = values.abs()
                direction = 1  # Flag errors above the normal range.
            else:
                direction = -1  # Flag signal/count values below the normal range.

            median = float(values.median())
            iqr = float(values.quantile(0.75) - values.quantile(0.25))
            # If IQR is zero, a strict comparison flags any departure in the
            # relevant direction from the median. This preserves the rule.
            limits[tractor][measurement] = (
                median + direction * config["iqr_multiplier"] * iqr
            )

        signal = rows["mean_cn0_dbhz"]
        signal_iqr = signal.quantile(0.75) - signal.quantile(0.25)
        limits[tractor]["severe_cn0"] = float(
            signal.median() - config["severe_iqr_multiplier"] * signal_iqr
        )

        signal_gap = rows["mean_cn0_dbhz"] - rows["min_cn0_dbhz"]
        gap_iqr = signal_gap.quantile(0.75) - signal_gap.quantile(0.25)
        limits[tractor]["signal_spread"] = float(
            signal_gap.median() + config["iqr_multiplier"] * gap_iqr
        )
    return limits

def create_flags(data, limits, config):
    """Add symptom columns without changing the caller's measurements."""
    data = data.copy()
    absolute_error = data["cross_track_error_m"].abs()
    # transform repeats each timestamp's fleet median on every tractor's row.
    data["fleet_error"] = absolute_error.groupby(data["ts"]).transform("median")

    tractor_tables = []
    for tractor, rows in data.groupby("tractor_id"):
        if tractor not in limits:
            raise ValueError(f"No development reference for {tractor}")
        rows = rows.copy()
        tractor_limits = limits[tractor]
        absolute_error = rows["cross_track_error_m"].abs()
        is_moving = rows["speed_mps"] > 0
        signal_gap = rows["mean_cn0_dbhz"] - rows["min_cn0_dbhz"]

        # Each comparison produces a whole column of True/False values.
        rows["low_cn0"] = rows["mean_cn0_dbhz"] < tractor_limits["mean_cn0_dbhz"]
        rows["weak_min_cn0"] = rows["min_cn0_dbhz"] < tractor_limits["min_cn0_dbhz"]
        rows["high_error"] = (
            (absolute_error > tractor_limits["cross_track_error_m"]) & is_moving
        )
        rows["few_sats"] = rows["num_sats"] < tractor_limits["num_sats"]
        rows["link_failure"] = ~rows["base_link_ok"].astype(bool)
        rows["fix_degraded"] = rows["fix_state"] != "RTK_FIXED"
        rows["severe_cn0"] = rows["mean_cn0_dbhz"] < tractor_limits["severe_cn0"]
        rows["signal_spread"] = signal_gap > tractor_limits["signal_spread"]
        rows["excess_error"] = rows["high_error"] & (
            absolute_error > config["fleet_error_ratio"] * rows["fleet_error"]
        )
        tractor_tables.append(rows)

    return pd.concat(tractor_tables).sort_values(["tractor_id", "ts"])

def find_intervals(rows, symptom, config):
    """Group one symptom's flags, allowing brief recoveries between flags.

    Missing samples always split intervals. Each symptom is handled separately
    so a short event inside a long event can still be classified on its own.
    """
    rows = rows.sort_values("ts")

    # diff compares each timestamp with the previous one. A gap in the input
    # starts a new continuity group; cumsum numbers those groups.
    sample_gap = rows["ts"].diff()
    missing_sample = sample_gap.ne(SECOND)
    continuity_group = missing_sample.cumsum()

    hits = rows.loc[rows[symptom], ["ts"]].copy()
    hits["continuity"] = continuity_group.loc[hits.index]
    if hits.empty:
        return []

    # Flags four seconds apart have THREE intervening unflagged seconds.
    maximum_hit_spacing = pd.Timedelta(seconds=config["gap_seconds"] + 1)
    recovery_too_long = hits["ts"].diff().gt(maximum_hit_spacing)
    crossed_missing_sample = hits["continuity"].diff().ne(0)
    starts_new_run = recovery_too_long | crossed_missing_sample
    hits["run"] = starts_new_run.cumsum()

    intervals = []
    for _, run in hits.groupby("run"):
        start = run["ts"].iloc[0]
        end = run["ts"].iloc[-1] + SECOND  # End is exclusive.
        duration_seconds = (end - start).total_seconds()
        flagged_seconds = len(run)
        density = flagged_seconds / duration_seconds

        enough_hits = flagged_seconds >= config["minimum_hits"]
        dense_enough = density >= config["minimum_density"]
        if enough_hits and dense_enough:
            intervals.append({
                "start": start,
                "end": end,
                "trigger": symptom,
                "flagged_seconds": flagged_seconds,
                "density": density,
            })
    return intervals

def align_feed(data, records, key, tolerance):
    """Attach the latest feed value at or before each telemetry timestamp.

    merge_asof matches nearby times instead of requiring identical timestamps.
    Values older than tolerance are left missing, rather than carried forever.
    """
    feed = pd.DataFrame(records)
    feed["ts"] = pd.to_datetime(feed["ts"], utc=True)
    return pd.merge_asof(
        data.sort_values("ts"), feed[["ts", key]].sort_values("ts"),
        on="ts", direction="backward", tolerance=pd.Timedelta(tolerance),
    )

def classify(rows, trigger, feeds, config, diagnostics=None):
    """Return competing cause scores and the observations used to compute them.

    Scores express support. These rules are a starting model for development,
    and different symptoms in overlapping intervals can yield different causes.
    """
    # True counts as 1 and False as 0: the mean is the fraction flagged.
    flag_fractions = rows[FLAGS].mean()
    duration_seconds = (rows["ts"].max() - rows["ts"].min()).total_seconds() + 1
    s4 = float(rows["s4"].median())
    kp = float(rows["kp"].median())
    elevated_s4 = np.isfinite(s4) and s4 >= config["s4_elevated"]
    elevated_kp = np.isfinite(kp) and kp >= config["kp_elevated"]
    # Regional RFI reports have uncertain locations. A generous spatial circle
    # and a time neighborhood provide support, without proving attribution.
    matching_rfi = []
    for report in feeds["rfi"]:
        timestamp = pd.Timestamp(report["ts"])
        earliest_report = rows["ts"].min() - pd.Timedelta(minutes=20)
        latest_report = rows["ts"].max() + pd.Timedelta(minutes=20)
        near_time = earliest_report <= timestamp <= latest_report
        lat = float(rows["lat"].median())
        # Approximate degrees of latitude/longitude as distances in kilometers.
        north_south_km = (lat - report["centerLat"]) * 111
        east_west_km = (
            (float(rows["lon"].median()) - report["centerLon"])
            * 111 * np.cos(np.radians(lat))
        )
        distance = np.hypot(north_south_km, east_west_km)
        if near_time and distance <= report["semiMajor_km"] + 2:
            matching_rfi.append(report)
    rfi_confidence = max((r["confidence"] for r in matching_rfi), default=0)
    if diagnostics is not None:
        diagnostics.update({
            "max_cross_track_error_m": float(rows["cross_track_error_m"].abs().max()),
            "matched_rfi": matching_rfi,
            "rfi_confidence": rfi_confidence,
            "elevated_s4": bool(elevated_s4),
            "elevated_kp": bool(elevated_kp),
        })
    low_signal_fraction = float(flag_fractions["low_cn0"])
    high_error_fraction = float(flag_fractions["high_error"])
    link_failure_fraction = float(flag_fractions["link_failure"])
    # Keep the existing weights: larger scores mean more support for a cause.
    scores = {
        "SPACE_WEATHER": 4 * elevated_s4 + elevated_kp + low_signal_fraction,
        "JAMMING": (
            3 * low_signal_fraction
            + 3 * rfi_confidence
            + float(flag_fractions["few_sats"])
            + (1 if not elevated_s4 else 0)
        ),
        "SPOOFING": (
            3 * high_error_fraction
            + 2 * (1 - low_signal_fraction)
            - 2 * link_failure_fraction
        ),
        "EQUIPMENT_FAULT": (
            6 * link_failure_fraction
            + 2 * float(flag_fractions["weak_min_cn0"]) * (1 - low_signal_fraction)
        ),
        "MULTIPATH": 0.0,
    }
    if duration_seconds <= config["short_event_seconds"]:
        scores["MULTIPATH"] = (
            4 * high_error_fraction
            + 2 * float(flag_fractions["fix_degraded"])
            + (1 - link_failure_fraction)
        )
    # The trigger is the symptom used to construct this candidate interval.
    # Add the existing bonuses for that symptom.
    if trigger == "link_failure":
        scores["EQUIPMENT_FAULT"] += 3
    if trigger == "high_error":
        scores["SPOOFING"] += 1
    if trigger == "severe_cn0":
        scores["JAMMING"] += 4
    if trigger == "excess_error":
        scores["SPOOFING"] += 4
    if trigger == "signal_spread":
        if duration_seconds <= config["short_event_seconds"]:
            scores["MULTIPATH"] += 6
        else:
            scores["EQUIPMENT_FAULT"] += 7
    for cause, score in scores.items():
        nonnegative_score = max(0, float(score))
        scores[cause] = round(nonnegative_score, 3)
    # Choose the cause with the highest score; these are not probabilities.
    winner = max(scores, key=scores.get)
    flag_descriptions = []
    for name, fraction in flag_fractions.items():
        flag_descriptions.append(f"{name}={fraction:.2f}")
    flag_summary = ", ".join(flag_descriptions)

    report_ids = []
    for report in matching_rfi:
        report_ids.append(report["id"])
    rfi_summary = ", ".join(report_ids) or "none nearby"

    median_signal = rows["mean_cn0_dbhz"].median()
    maximum_error = rows["cross_track_error_m"].abs().max()
    median_affected = rows["fleet_affected"].median()
    evidence = [
        f"Trigger {trigger}; C/N0 median {median_signal:.2f} dB-Hz; "
        f"path error maximum {maximum_error:.3f} m.",
        "Flag fractions " + flag_summary,
        f"Kp median {kp:.2f}; S4 median {s4:.3f}; RFI reports " + rfi_summary,
        f"Median simultaneously affected tractors {median_affected:.0f}.",
    ]
    return winner, scores, evidence

def detect(data, episodes, feeds, config, diagnostics=None):
    """Run the detection pipeline and return merged events and learned limits."""
    # 1. Learn normal measurements and mark unusual measurements.
    limits = learn_limits(data, episodes, config)
    flagged = create_flags(data, limits, config)

    # 2. Add fleet context and the most recent available environmental readings.
    flagged["any_flag"] = flagged[FLAGS].any(axis=1)
    flagged["fleet_affected"] = flagged.groupby("ts")["any_flag"].transform("sum")
    flagged = align_feed(flagged, feeds["kp"], "kp", "3h")
    flagged = align_feed(flagged, feeds["s4"], "s4", "20min")

    # 3. Find and classify intervals for each tractor and each symptom.
    candidates = []
    operational = flagged.loc[flagged["speed_mps"] > 0]
    for tractor, rows in operational.groupby("tractor_id"):
        for symptom in FLAGS:
            interval_config = config
            if symptom == "link_failure":
                # Link failures can be intermittent: allow longer recoveries.
                interval_config = config.copy()
                interval_config.update({
                    "gap_seconds": 60,
                    "minimum_hits": 3,
                    "minimum_density": 0.05,
                })

            intervals = find_intervals(rows, symptom, interval_config)
            for interval in intervals:
                inside_interval = (
                    (rows["ts"] >= interval["start"])
                    & (rows["ts"] < interval["end"])
                )
                segment = rows.loc[inside_interval]
                candidate_diagnostics = {}
                cause, scores, evidence = classify(segment, symptom, feeds, config, candidate_diagnostics)

                candidate = interval.copy()
                candidate.update({
                    "tractors": [tractor],
                    "class": cause,
                    "scores": scores,
                    "evidence": evidence,
                    # Additive diagnostics only: no changes to flags or scores.
                    "diagnostics": {
                        **candidate_diagnostics,
                        "interval_settings": interval_config.copy(),
                        "flag_fractions": {name: float(segment[name].mean()) for name in FLAGS},
                        "kp_median": float(segment["kp"].median()),
                        "s4_median": float(segment["s4"].median()),
                    },
                })
                candidates.append(candidate)

    # Intermittent antenna symptoms need persistence across recoveries, rather
    # than treating each dip as an attack. Keep the short-burst path above too.
    for tractor, rows in operational.groupby("tractor_id"):
        antenna_config = {**config,
            "gap_seconds": config.get("antenna_gap_seconds", 60),
            "minimum_hits": config.get("antenna_minimum_hits", 30),
            "minimum_density": config.get("antenna_minimum_density", .15)}
        for interval in find_intervals(rows, "signal_spread", antenna_config):
            if (interval["end"] - interval["start"]).total_seconds() < config.get("antenna_minimum_seconds", 120):
                continue
            segment = rows.loc[(rows.ts >= interval["start"]) & (rows.ts < interval["end"])]
            # Regional mean-signal collapse is not antenna-specific evidence.
            if segment.low_cn0.mean() >= .5:
                continue
            candidate_diagnostics = {}
            cause, scores, evidence = classify(segment, "signal_spread", feeds, config, candidate_diagnostics)
            candidates.append({**interval, "tractors": [tractor], "class": cause,
                "scores": scores, "evidence": evidence,
                "diagnostics": {**candidate_diagnostics,
                    "interval_settings": antenna_config,
                    "construction": "persistent intermittent signal spread",
                    "flag_fractions": {f: float(segment[f].mean()) for f in FLAGS},
                    "kp_median": float(segment.kp.median()),
                    "s4_median": float(segment.s4.median())}})

    # 4. Reject weak or single-symptom hypotheses BEFORE grouping.
    # A raw signal_spread trigger automatically earns 6 MULTIPATH points.
    # Without an independent degradation signal that bonus is NOT evidence
    # of an operational multipath event; it produced ~1,760 noisy reports.
    retained_by_score = []
    suppressed = []
    rejection_counts = {}
    for candidate in candidates:
        cause = candidate["class"]
        fractions = candidate["diagnostics"]["flag_fractions"]
        score = float(candidate["scores"][cause])
        reason = None
        if score < config["minimum_cause_score"]:
            reason = (f"Top score {score:.3f} below required "
                      f"{config['minimum_cause_score']:.3f}")
        elif cause == "MULTIPATH":
            if not multipath_supported(candidate, config):
                reason = ("Multipath requires signal spread AND an independent "
                          "guidance-error or GNSS fix-degradation symptom")
        if reason:
            reason_name = "weak_score" if score < config["minimum_cause_score"] else "unsupported_multipath"
            rejection_counts[reason_name] = rejection_counts.get(reason_name, 0) + 1
            # Keep a bounded audit sample to avoid multi-megabyte diagnostic files.
            if len(suppressed) < 400:
                suppressed.append({**candidate, "suppression_reason": reason})
        else:
            retained_by_score.append(candidate)
    candidates = retained_by_score
    retained, duplicates = resolve_short_duplicates(candidates, config)
    rejection_counts["duplicate_or_context"] = len(duplicates)
    suppressed.extend(duplicates[:max(0, 400 - len(suppressed))])
    supported = []
    tractor_rows = {tractor: rows.sort_values("ts").set_index("ts", drop=False)
                    for tractor, rows in operational.groupby("tractor_id")}
    for candidate in retained:
        fractions = candidate["diagnostics"]["flag_fractions"]
        duration = (candidate["end"] - candidate["start"]).total_seconds()
        reason = None
        if candidate["class"] == "SPOOFING" and duration < config.get("antenna_minimum_seconds", 120):
            rows = tractor_rows[candidate["tractors"][0]]
            half = pd.Timedelta(seconds=config.get("spoof_context_seconds", 300) / 2)
            midpoint = candidate["start"] + (candidate["end"] - candidate["start"]) / 2
            context = rows.iloc[rows.index.searchsorted(midpoint-half):rows.index.searchsorted(midpoint+half)]
            support = float(context.excess_error.mean())
            link_failure = float(context.link_failure.mean())
            candidate["diagnostics"]["spoof_context"] = {
                "start": midpoint-half, "end": midpoint+half,
                "excess_error_fraction": support, "link_failure_fraction": link_failure}
            if support < config.get("spoof_minimum_context_fraction", .5) or link_failure > .05:
                reason = "Brief spoofing hypothesis lacks persistent fleet-relative error in its context, or coincides with a correction-link failure."
        elif (candidate["class"] == "JAMMING" and duration <= config["short_event_seconds"]
              and not fractions["severe_cn0"] and not candidate["diagnostics"]["rfi_confidence"]):
            reason = "Brief jamming hypothesis lacks severe signal collapse or matching RFI corroboration."
        elif (candidate["class"] == "EQUIPMENT_FAULT" and candidate["trigger"] == "signal_spread"
              and fractions["low_cn0"] >= .5 and not fractions["link_failure"]):
            reason = "Signal spread during broad mean-signal collapse is insufficient independent equipment-fault evidence."
        if reason:
            rejection_counts["duplicate_or_context"] = rejection_counts.get("duplicate_or_context", 0) + 1
            if len(suppressed) < 400:
                suppressed.append({**candidate, "suppression_reason": reason})
        else:
            supported.append(candidate)
    retained = supported
    continuity = {tractor: rows.ts.sort_values().astype("int64").to_numpy()
                  for tractor, rows in operational.groupby("tractor_id")}
    events = merge_candidates(retained, config.get("merge_gap_seconds", 60), continuity)
    # A pattern repeated on the entire fleet every ~6 minutes is not
    # independently convincing evidence of hundreds of distinct multipath
    # incidents. This is determined from detector outputs, not episode labels.
    events, multipath_background = group_recurrent_multipath(events, config)
    if diagnostics is not None:
        diagnostics["multipath_background"] = {
            k: v for k, v in multipath_background.items() if k != "suppressed_sample"
        }
        diagnostics.update(raw_candidate_count=len(candidates) + sum(
                               count for label, count in rejection_counts.items()
                               if label in ("weak_score", "unsupported_multipath")),
                           suppressed_candidates=suppressed,
                           suppressed_sample_only=True,
                           suppression_counts=rejection_counts,
                           retained_candidate_count=len(retained),
                           policy="operational-grouping-v3")
    return events, limits

def multipath_supported(candidate, config):
    """Require two independent measurements, not the trigger's +6 bonus alone."""
    f = candidate["diagnostics"]["flag_fractions"]
    return (f["signal_spread"] >= config["multipath_min_signal_spread"]
            and (f["high_error"] >= config["multipath_min_high_error"]
                 or f["fix_degraded"] >= config["multipath_min_fix_degraded"]))


def resolve_short_duplicates(candidates, config):
    """Suppress alternate attack labels for well-supported brief multipath.

    Requires >=80% temporal coverage by same-tractor multipath, signal spread
    in >=50% of the candidate, and <50% fleet-relative excess-error flags.
    Severe signal collapse and sustained/independent errors stay separate.
    Original scores and suppression reasons remain in diagnostic output.
    """
    multipath = [c for c in candidates if c["class"] == "MULTIPATH"]
    retained, suppressed = [], []
    for candidate in candidates:
        duration = (candidate["end"] - candidate["start"]).total_seconds()
        fractions = candidate.get("diagnostics", {}).get("flag_fractions", {})
        eligible = (candidate["class"] == "SPOOFING"
                    and candidate["trigger"] in ("high_error", "weak_min_cn0")
                    and duration <= config["short_event_seconds"]
                    and fractions.get("signal_spread", 0) >= .5
                    and fractions.get("excess_error", 1) < .5)
        spans = []
        if eligible:
            for other in multipath:
                if (set(other["tractors"]) & set(candidate["tractors"])
                        and other["start"] < candidate["end"]
                        and other["end"] > candidate["start"]):
                    spans.append((max(candidate["start"], other["start"]),
                                  min(candidate["end"], other["end"])))
        covered, previous_end = 0., None
        for start, end in sorted(spans):
            covered += max(0., (end - max(start, previous_end or start)).total_seconds())
            previous_end = max(end, previous_end or end)
        if eligible and covered / duration >= .8:
            suppressed.append({**candidate,
                "suppression_reason": "Short signal-spread burst overlaps same-tractor multipath by at least 80%; no sustained or majority excess-error support.",
                "multipath_overlap_fraction": covered / duration})
        else:
            retained.append(candidate)
    return retained, suppressed


def merge_candidates(candidates, gap_seconds=0, continuity=None):
    """Merge overlapping candidates of the same cause, including across tractors.

    Different causes stay separate. Near fragments join only when they share
    a tractor and its operational samples are continuous across the gap.
    Original members retain the true evidence boundaries; gaps are not flags.
    """
    merged = []
    for cause in ACTIONS:
        items = sorted((c for c in candidates if c["class"] == cause),
                       key=lambda c: c["start"])
        parents = list(range(len(items)))

        def root(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        active = []
        for index, candidate in enumerate(items):
            active = [j for j in active if
                      (candidate["start"] - items[j]["end"]).total_seconds() <= gap_seconds]
            for j in active:
                other = items[j]
                # Recurrent multipath is a brief per-machine disturbance.
                # Fleet-wide unions or recovery-gap grouping obscure the
                # representative 17-second label and its affected tractor.
                overlaps = (candidate["start"] < other["end"] and
                            (cause != "MULTIPATH" or bool(set(candidate["tractors"]) & set(other["tractors"]))))
                near = False
                gap = (candidate["start"] - other["end"]).total_seconds()
                if cause != "MULTIPATH" and not overlaps and gap_seconds > 0 and 0 <= gap <= gap_seconds:
                    for tractor in set(other["tractors"]) & set(candidate["tractors"]):
                        samples = continuity.get(tractor) if continuity is not None else None
                        if samples is None and continuity is not None:
                            continue
                        first = (other["end"] - SECOND).value
                        last = candidate["start"].value
                        if samples is None or (np.searchsorted(samples, last, side="right") - np.searchsorted(samples, first)) == int((last-first)/1e9)+1:
                            near = True
                            break
                if overlaps or near:
                    parents[root(index)] = root(j)
            active.append(index)

        components = {}
        for index, candidate in enumerate(items):
            components.setdefault(root(index), []).append(candidate)
        for members in components.values():
            merged.append({"start": min(m["start"] for m in members),
                "end": max(m["end"] for m in members), "class": cause,
                "tractors": sorted({t for m in members for t in m["tractors"]}),
                "members": members})
    return sorted(merged, key=lambda event: event["start"])




def group_recurrent_multipath(events, config):
    """Consolidate short multipath bursts into evidence-backed reporting windows.

    Original candidate intervals remain inside ``members``. The envelope is a
    *reporting window*, NOT a claim that multipath was active in all intervening
    seconds. Different causes never merge. A maximum window prevents a train of
    bursts from being reported as a single day-long disturbance.

    This rule is independent of labels and does not target a fixed event count.
    """
    multipath = sorted((e for e in events if e["class"] == "MULTIPATH"),
                       key=lambda e: (e["start"], e["end"]))
    other = [e for e in events if e["class"] != "MULTIPATH"]
    gap = pd.Timedelta(seconds=config["multipath_group_gap_seconds"])
    max_span = pd.Timedelta(seconds=config["multipath_max_group_seconds"])
    clusters = []
    current = []
    earliest = latest = None

    def finish():
        if not current:
            return
        members = [m for event in current for m in event["members"]]
        clusters.append({
            "start": min(m["start"] for m in members),
            "end": max(m["end"] for m in members),
            "class": "MULTIPATH",
            "tractors": sorted({t for m in members for t in m["tractors"]}),
            "members": members,
            "reporting_note": ("Grouped intermittent multipath evidence; "
                               "interval envelope includes unflagged gaps"),
            "burst_count": len(current),
        })

    for event in multipath:
        start, end = event["start"], event["end"]
        # Do not join signals separated by a substantial quiet period.
        # Never let an accumulation of small gaps create an unbounded envelope.
        if current and (start > latest + gap or end - earliest > max_span):
            finish()
            current = []
            earliest = latest = None
        if not current:
            earliest = start
            latest = end
        else:
            latest = max(latest, end)
        current.append(event)
    finish()
    return sorted(other + clusters, key=lambda e: e["start"]), {
        "original_multipath_events": len(multipath),
        "grouped_multipath_events": len(clusters),
        "gap_seconds": gap.total_seconds(),
        "max_group_seconds": max_span.total_seconds(),
        "note": "Reports aggregate bursts; envelope gaps are not asserted as detections",
    }


def evaluate(events, episodes):
    """Print development coverage and duration outside matching labels.

    These labels were also used when selecting reference measurements, so this
    is not an independent evaluation. Overlap alone is not a boundary match.
    """
    def covered_seconds(intervals):
        """Count total covered time, counting overlapping portions only once."""
        total = 0
        previous_end = None
        for start, end in sorted(intervals):
            if previous_end is None or start >= previous_end:
                total += (end - start).total_seconds()
            elif end > previous_end:
                total += (end - previous_end).total_seconds()

            if previous_end is None:
                previous_end = end
            else:
                previous_end = max(previous_end, end)
        return total

    for label in episodes:
        start = pd.Timestamp(label["start"])
        end = pd.Timestamp(label["end"])
        coverage = []
        for tractor in label["tractors"]:
            overlaps = []
            for event in events:
                if event["class"] != label["class"]:
                    continue
                for member in event["members"]:
                    if tractor not in member["tractors"]:
                        continue
                    if member["start"] < end and member["end"] > start:
                        overlaps.append((
                            max(start, member["start"]),
                            min(end, member["end"]),
                        ))
            label_duration = (end - start).total_seconds()
            coverage.append(covered_seconds(overlaps) / label_duration)
        print(f"{label['id']} {label['class']}: mean affected-tractor coverage {np.mean(coverage):.1%}")

    for cause in ACTIONS:
        predicted_seconds = 0
        matching_seconds = 0
        for tractor in ("T1", "T2", "T3"):
            spans = []
            for event in events:
                if event["class"] != cause:
                    continue
                for member in event["members"]:
                    if tractor in member["tractors"] and member["start"] < EVAL_START:
                        spans.append((member["start"], min(member["end"], EVAL_START)))
            predicted_seconds += covered_seconds(spans)

            overlaps = []
            for start, end in spans:
                for label in episodes:
                    if label["class"] != cause or tractor not in label["tractors"]:
                        continue
                    label_start = pd.Timestamp(label["start"])
                    label_end = pd.Timestamp(label["end"])
                    if start < label_end and end > label_start:
                        overlaps.append((max(start, label_start), min(end, label_end)))
            matching_seconds += covered_seconds(overlaps)
        print(f"{cause}: predicted tractor-seconds {predicted_seconds:.0f}, "
              f"outside matching labels {predicted_seconds - matching_seconds:.0f}")

def self_test():
    """Independent artificial flag patterns, including gaps and boundaries."""
    config = {**DEFAULTS, "minimum_hits": 3, "gap_seconds": 1}
    times = pd.date_range("2026-05-04", periods=12, freq="s", tz="UTC")
    cases = [
        ("quiet", [0]*12, []),
        ("isolated noise", [0,0,1,0,0,0,0,0,0,0,0,0], []),
        ("sustained", [0,0,1,1,1,1,0,0,0,0,0,0], [(2,6)]),
        ("brief recovery", [0,1,1,0,1,1,0,0,0,0,0,0], [(1,6)]),
        ("separate episodes", [1,1,1,0,0,0,1,1,1,0,0,0], [(0,3),(6,9)]),
        ("end of file", [0]*9+[1]*3, [(9,12)]),
    ]
    for name, values, expected in cases:
        rows = pd.DataFrame({"ts": times, "flag": np.array(values, dtype=bool)})
        result = find_intervals(rows, "flag", config)
        actual = []
        for interval in result:
            start_offset = int((interval["start"] - times[0]).total_seconds())
            end_offset = int((interval["end"] - times[0]).total_seconds())
            actual.append((start_offset, end_offset))
        assert actual == expected, (name, actual, expected)
        print("PASS", name)
    rows = pd.DataFrame({"ts": times.delete(5), "flag": True})
    result = find_intervals(rows, "flag", config)
    assert len(result) == 2
    print("PASS missing sample splits intervals")
    # Artificial measurements exercise flags, intervals, and cause scoring.
    limits = {"T1": {"mean_cn0_dbhz": 40, "min_cn0_dbhz": 34,
                     "cross_track_error_m": .1, "num_sats": 8,
                     "severe_cn0": 25, "signal_spread": 8}}
    scenarios = [
        ("JAMMING", "severe_cn0", 120, {"mean_cn0_dbhz": 22, "min_cn0_dbhz": 18, "num_sats": 5}),
        ("SPACE_WEATHER", "low_cn0", 120, {"mean_cn0_dbhz": 32, "min_cn0_dbhz": 27, "s4": .7, "kp": 7}),
        ("SPOOFING", "high_error", 120, {"cross_track_error_m": 3}),
        ("EQUIPMENT_FAULT", "link_failure", 120, {"base_link_ok": False}),
        ("MULTIPATH", "signal_spread", 17, {"min_cn0_dbhz": 25, "cross_track_error_m": .2}),
    ]
    for expected, trigger, length, changes in scenarios:
        values = {"tractor_id": "T1", "mean_cn0_dbhz": 45., "min_cn0_dbhz": 40.,
                  "cross_track_error_m": .02, "num_sats": 12, "speed_mps": 4.5,
                  "base_link_ok": True, "fix_state": "RTK_FIXED", "lat": 46.9,
                  "lon": -97.2, "s4": .05, "kp": 2., "fleet_affected": 1}
        values.update(changes)
        rows = pd.DataFrame(values, index=range(length))
        rows["ts"] = pd.date_range("2026-05-04", periods=length, freq="s", tz="UTC")
        flagged = create_flags(rows, limits, DEFAULTS)
        intervals = find_intervals(flagged, trigger, DEFAULTS)
        assert len(intervals) == 1, expected
        winner, _, _ = classify(flagged, trigger, {"rfi": []}, DEFAULTS)
        assert winner == expected, (expected, winner)
        print("PASS artificial", expected)
    # Regression: a large signal-spread score alone must not be a multipath event.
    suspect = {"class": "MULTIPATH", "scores": {"MULTIPATH": 7.0},
               "diagnostics": {"flag_fractions": {"signal_spread": 1.0,
                   "high_error": 0.0, "fix_degraded": 0.0}}}
    assert not multipath_supported(suspect, config)
    suspect["diagnostics"]["flag_fractions"]["high_error"] = 0.7
    assert multipath_supported(suspect, config)
    print("14 synthetic checks passed (12 original + 2 multipath gating checks).")




# Embedded submission validation and development verification: no project-local imports.
CLASSES = {"SPACE_WEATHER", "JAMMING", "SPOOFING", "EQUIPMENT_FAULT", "MULTIPATH"}


TRACTORS = {"T1", "T2", "T3"}


HANDLE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")


REPO_URL = re.compile(r"^https://github\.com/[A-Za-z0-9-]{1,39}/[A-Za-z0-9._-]{1,100}/?$")


PLACEHOLDERS = ("your-github-username", "your-solution-repo", "Optional name")


PRIVATE_FIELDS = ("email", "student_id", "studentid", "phone", "ndsu_id")


MAX_DETECTIONS = 500


def validate(sub):
    problems = []
    if not isinstance(sub, dict):
        return ["submission must be a JSON object"]

    if "candidate" in sub:
        problems.append("remove the old 'candidate' block; use 'participant' with your GitHub username "
                        "(never put your email or student ID in a public submission)")
    part = sub.get("participant", {})
    if not isinstance(part, dict):
        problems.append("participant must be an object")
        part = {}
    for key in part:
        if str(key).lower().replace("-", "_") in PRIVATE_FIELDS:
            problems.append(f"participant.{key} must not be included: submissions are public")
    handle = part.get("github_handle", "")
    if not isinstance(handle, str) or not HANDLE.match(handle) or handle in PLACEHOLDERS:
        problems.append("participant.github_handle must be your GitHub username")
    name = part.get("display_name", "")
    if name is not None and (not isinstance(name, str) or len(name) > 60 or "@" in name
                             or name.startswith(PLACEHOLDERS)):
        problems.append("participant.display_name is optional; if given, use up to 60 characters and no email address")
    url = sub.get("repository_url", "")
    if not isinstance(url, str) or not REPO_URL.match(url) or any(p in url for p in PLACEHOLDERS):
        problems.append("repository_url must be your solution repository, https://github.com/<you>/<repo>")
    elif isinstance(handle, str) and handle and url.split("/")[3].lower() != handle.lower():
        problems.append("repository_url should be a repository you own (github.com/<your username>/...)")

    dets = sub.get("detections")
    if not isinstance(dets, list) or not dets:
        problems.append("detections must be a non-empty list")
        dets = []
    elif len(dets) > MAX_DETECTIONS:
        problems.append(f"detections has {len(dets)} entries; the limit is {MAX_DETECTIONS}")
        dets = []

    for i, d in enumerate(dets):
        tag = f"detections[{i}]"
        if not isinstance(d, dict):
            problems.append(f"{tag} must be an object")
            continue
        try:
            a = datetime.fromisoformat(d["start"].replace("Z", "+00:00"))
            b = datetime.fromisoformat(d["end"].replace("Z", "+00:00"))
            if a.utcoffset() is None or b.utcoffset() is None:
                raise ValueError("Timezone offset required")
            if not (datetime.fromisoformat("2026-05-06T00:00:00+00:00") <= a < datetime.fromisoformat("2026-05-07T00:00:00+00:00")):
                problems.append(f"{tag}.start outside the evaluation window")
            if b <= a:
                problems.append(f"{tag}: end must be after start")
            if b > datetime.fromisoformat("2026-05-07T00:00:00+00:00"):
                problems.append(f"{tag}.end outside the evaluation window")
        except Exception:
            problems.append(f"{tag}: start/end must be ISO-8601 timestamps with a timezone")
        if not isinstance(d.get("class"), str) or d["class"] not in CLASSES:
            problems.append(f"{tag}.class must be one of {sorted(CLASSES)}")
        if not isinstance(d.get("action"), str) or d["action"] not in set(ACTIONS.values()):
            problems.append(f"{tag}.action must be one of {sorted(ACTIONS.values())}")
        trs = d.get("tractors", [])
        if (not isinstance(trs, list) or not trs
                or not all(isinstance(t, str) and t in TRACTORS for t in trs)
                or len(set(trs)) != len(trs)):
            problems.append(f"{tag}.tractors must be a non-empty list of distinct T1/T2/T3 IDs")
        c = d.get("confidence")
        if (isinstance(c, bool) or not isinstance(c, (int, float))
                or not math.isfinite(c) or not 0.0 <= c <= 1.0):
            problems.append(f"{tag}.confidence must be a number in [0,1]")
        ev = d.get("evidence", [])
        if not isinstance(ev, list) or not ev or not all(
                isinstance(s, str) and s.strip() for s in ev):
            problems.append(f"{tag}.evidence must be a non-empty list of strings")
    return problems


def development_report(events, labels):
    labels = [l for l in labels if pd.Timestamp(l['end']) <= EVAL_START]
    edges = {}
    for event_id, event in enumerate(events):
        start, end = pd.Timestamp(event['start']), pd.Timestamp(event['end'])
        if start >= EVAL_START:
            continue
        end = min(end, EVAL_START)
        for label_id, label in enumerate(labels):
            if event['class'] != label['class']:
                continue
            a,b = pd.Timestamp(label['start']), pd.Timestamp(label['end'])
            overlap = max(0., (min(end,b)-max(start,a)).total_seconds())
            union = (max(end,b)-min(start,a)).total_seconds()
            tiou = overlap/union
            predicted = {t for m in event['members'] if pd.Timestamp(m['start']) < EVAL_START for t in m['tractors']}
            expected = set(label['tractors'])
            tractor_iou = len(predicted & expected)/len(predicted | expected)
            if tiou >= .30 and tractor_iou >= .50:
                edges.setdefault(event_id, []).append((label_id, overlap, tiou, tractor_iou))
    # Small labeled set (12): dynamic programming gives a deterministic exact
    # one-to-one assignment without another dependency or greedy mis-matches.
    states = {0: (0., [])}
    for event_id, choices in edges.items():
        updated = states.copy()
        for mask,(score,pairs) in states.items():
            for label_id,overlap,tiou,tractor_iou in choices:
                if mask & (1 << label_id):
                    continue
                new_mask = mask | (1 << label_id)
                value = (score+overlap,pairs+[(label_id,event_id,tiou,tractor_iou)])
                if new_mask not in updated or value[0] > updated[new_mask][0]:
                    updated[new_mask] = value
        states = updated
    best_mask = max(states,key=lambda mask:(mask.bit_count(),states[mask][0]))
    matched = {label_id:(event_id,tiou,tractor_iou) for label_id,event_id,tiou,tractor_iou in states[best_mask][1]}
    rows=[]
    for label_id,label in enumerate(labels):
        match = matched.get(label_id)
        event = events[match[0]] if match else None
        rows.append({'Episode':label['id'],'Cause':label['class'],
            'Status':'Matched' if match else 'Missed',
            'Prediction':match[0] if match else None,
            'Time IoU':round(match[1],3) if match else 0.,
            'Tractor IoU':round(match[2],3) if match else 0.,
            'Start offset (s)':(pd.Timestamp(event['start'])-pd.Timestamp(label['start'])).total_seconds() if event else None,
            'End offset (s)':(pd.Timestamp(event['end'])-pd.Timestamp(label['end'])).total_seconds() if event else None})
    return rows


def build_submission(events, github_handle, repo_url):
    """Clip detector events to evaluation and validate the public deliverable."""

    detections = []
    for event in events:
        start = max(pd.Timestamp(event["start"]), EVAL_START)
        end = min(pd.Timestamp(event["end"]), EVAL_END)
        if start >= end:
            continue
        members = [member for member in event["members"]
                   if pd.Timestamp(member["end"]) > start
                   and pd.Timestamp(member["start"]) < end]
        if not members:
            raise ValueError("Evaluation event has no overlapping member evidence.")
        evidence = list(dict.fromkeys(
            item for member in members for item in member["evidence"]))
        if event["class"] == "MULTIPATH" and event.get("reporting_note"):
            evidence.insert(0, event["reporting_note"] +
                            f"; {event.get('burst_count', len(members))} detected bursts in envelope")
        detections.append({
            "start": start.isoformat(), "end": end.isoformat(),
            "class": event["class"],
            "tractors": sorted({tractor for member in members
                                for tractor in member["tractors"]}),
            "action": ACTIONS[event["class"]],
            "confidence": 0.5,
            "evidence": ["Heuristic attribution; confidence is uncalibrated."] + evidence,
        })
    if len(detections) > MAX_DETECTIONS:
        operational = [d for d in detections if d["class"] != "MULTIPATH"]
        recurring = sorted((d for d in detections if d["class"] == "MULTIPATH"),
                           key=lambda d: (d["start"], d["tractors"]))
        remaining = MAX_DETECTIONS - len(operational)
        if remaining >= 0:
            indices = np.linspace(0, len(recurring) - 1, remaining, dtype=int)
            detections = sorted(operational + [recurring[i] for i in indices],
                                key=lambda d: (d["start"], d["class"], d["tractors"]))
    result = {
        "participant": {"github_handle": github_handle},
        "repository_url": repo_url, "detections": detections,
    }
    problems = validate(result)
    if problems:
        raise ValueError("Invalid submission:\n  - " + "\n  - ".join(problems))
    return result


def development_scores(events, report):
    """Show measured winning-cause support for one-to-one matched events only."""
    records = []
    for row in report:
        if row["Status"] != "Matched":
            continue
        event = events[row["Prediction"]]
        scores = [float(member["scores"][event["class"]]) for member in event["members"]]
        records.append({"episode": row["Episode"], "cause": event["class"],
                        "score": max(scores), "time_iou": row["Time IoU"],
                        "tractor_iou": row["Tractor IoU"]})
    return sorted(records, key=lambda item: item["episode"])


def main():
    """Run independent measurement-based detection; labels only assess/calibrate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    parser.add_argument("--config", type=Path, help="Optional JSON overrides for DEFAULTS")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--no-calibration", action="store_true",
                        help="Use fixed minimum_cause_score (3.0 by default)")
    parser.add_argument("--score-margin", type=float, default=0.75,
                        help="Subtract this many points from the weakest matched development event")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--github-handle")
    parser.add_argument("--repo-url")
    parser.add_argument("--submission-output", type=Path, default=Path("submission.json"))
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if bool(args.github_handle) != bool(args.repo_url):
        parser.error("Supply both --github-handle and --repo-url.")
    if args.github_handle and args.output.resolve() == args.submission_output.resolve():
        parser.error("--output and --submission-output must be different files.")
    config = DEFAULTS.copy()
    if args.config:
        overrides = json.loads(args.config.read_text())
        unknown = set(overrides) - set(config)
        if unknown:
            parser.error(f"Unknown configuration setting(s): {sorted(unknown)}")
        config.update(overrides)
    for key in ("minimum_cause_score", "multipath_min_signal_spread",
                "multipath_min_high_error", "multipath_min_fix_degraded",
                "multipath_min_cross_track_error_m",
                "multipath_period_min_seconds", "multipath_period_max_seconds",
                "multipath_period_tolerance_seconds", "multipath_sync_tolerance_seconds",
                "multipath_group_gap_seconds", "multipath_max_group_seconds"):
        if not isinstance(config[key], (int,float)) or not np.isfinite(config[key]) or config[key] < 0:
            parser.error(f"{key} must be nonnegative and finite")
    if config["multipath_max_group_seconds"] <= 0:
        parser.error("multipath_max_group_seconds must be positive")
    if (config["multipath_period_max_seconds"] <= config["multipath_period_min_seconds"]
            or config["multipath_period_min_pairs"] < 1
            or config["multipath_peer_tractors"] < 0):
        parser.error("Invalid multipath recurrence settings")
    if not np.isfinite(args.score_margin) or args.score_margin < 0:
        parser.error("--score-margin must be a nonnegative finite number")
    if config["gap_seconds"] < 0 or config["merge_gap_seconds"] < 0 or config["minimum_hits"] < 1 or not 0 < config["minimum_density"] <= 1:
        parser.error("Invalid interval settings")
    try:
        data, episodes, feeds = load_data(args.data)
    except Exception as exc:
        parser.error(f"Data loading failed: {exc}")

    calibration = {"method": "no-calibration"}
    if not args.no_calibration:
        # Important: detection NEVER uses labeled event positions/classes.
        # The labels are used here only AFTER independent detection has finished.
        print("First pass: detect unfiltered events, then verify 12 labeled development episodes...", flush=True)
        first_config = {**config, "minimum_cause_score": 0.0}
        first_events, _ = detect(data, episodes, feeds, first_config)
        initial_report = development_report(first_events, episodes)
        scores = development_scores(first_events, initial_report)
        for label in initial_report:
            row = next((r for r in scores if r["episode"] == label["Episode"]), None)
            if row:
                print(f"  {row['episode']}: {row['cause']} score={row['score']:.3f} "
                      f"timeIoU={row['time_iou']:.3f} tractorIoU={row['tractor_iou']:.3f}", flush=True)
            else:
                print(f"  {label['Episode']}: NOT MATCHED", flush=True)
        if scores:
            weakest = min(scores, key=lambda s: s["score"])
            config["minimum_cause_score"] = max(0., weakest["score"] - args.score_margin)
            print(f"Derived minimum score: {config['minimum_cause_score']:.3f} "
                  f"(lowest matched {weakest['episode']}={weakest['score']:.3f}, minus {args.score_margin:.2f})", flush=True)
            calibration = {"method": "weakest-matched-development-minus-margin",
                           "first_pass_matched": len(scores),
                           "total_development": len(initial_report),
                           "matched_scores": scores, "weakest_matched": weakest,
                           "score_margin": args.score_margin,
                           "derived_threshold": config["minimum_cause_score"]}
        else:
            print("WARNING: No development labels matched; use fixed 3.0 cutoff instead.", flush=True)
            calibration = {"method": "fallback-fixed", "first_pass_matched": 0,
                           "derived_threshold": config["minimum_cause_score"]}
    else:
        print(f"Using fixed minimum score {config['minimum_cause_score']:.3f}.", flush=True)

    print("Final pass: apply score cutoff and independent corroboration requirements...", flush=True)
    diagnostics = {}
    events, limits = detect(data, episodes, feeds, config, diagnostics)
    report = development_report(events, episodes)
    n_matched = sum(row["Status"] == "Matched" for row in report)
    evaluation_events = [e for e in events if e["end"] > EVAL_START and e["start"] < EVAL_END]
    from collections import Counter
    summary = {"development_matched": n_matched, "development_total": len(report),
               "evaluation_event_count": len(evaluation_events),
               "evaluation_by_cause": dict(Counter(e["class"] for e in evaluation_events)),
               "full_diagnostic_event_count": len(events)}
    print(f"Development match: {n_matched}/{len(report)} (strict cause/time/tractor one-to-one match)", flush=True)
    for row in report:
        print(f"  {row['Episode']}: {row['Status']} "
              f"timeIoU={row['Time IoU']:.3f} tractorIoU={row['Tractor IoU']:.3f}", flush=True)
    print(f"May 6 evaluation predictions: {len(evaluation_events)} "
          f"(not necessarily the 8 hidden ground-truth episodes)", flush=True)
    print(f"By class: {summary['evaluation_by_cause']}", flush=True)
    print(f"Suppressed hypotheses: {diagnostics.get('suppression_counts', {})}", flush=True)
    print(f"Multipath background: {diagnostics.get('multipath_background', {})}", flush=True)
    if n_matched < len(report):
        print("WARNING: Some development events were missed. Do not report 12/12 accuracy.", flush=True)
    diagnostic_results = {"detector_version": "operational-grouping-v3",
        "pipeline_diagnostics": diagnostics, "settings": config,
        "learned_limits": limits, "events": events,
        "development_verification": report, "score_calibration": calibration,
        "run_summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(diagnostic_results, default=str, indent=2))
    print(f"Wrote {len(events)} diagnostic events to {args.output}", flush=True)
    if args.validate:
        evaluate(events, episodes)
    if args.github_handle:
        try:
            submission = build_submission(events, args.github_handle, args.repo_url)
        except ValueError as exc:
            parser.error(str(exc))
        args.submission_output.parent.mkdir(parents=True, exist_ok=True)
        args.submission_output.write_text(json.dumps(submission, indent=2) + "\n")
        print(f"Wrote {len(submission['detections'])} validated May 6 submissions to {args.submission_output}", flush=True)


if __name__ == "__main__":
    main()
