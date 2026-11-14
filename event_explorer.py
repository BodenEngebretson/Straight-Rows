#!/usr/bin/env python3
"""Inspect Straight Rows telemetry. Install pandas and matplotlib first."""

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def locate(root, filename):
    """Accept both the competition directory and a flat folder of downloads."""
    for folder in (root, root / "farm", root / "sda"):
        candidate = folder / filename
        if candidate.is_file():
            return candidate
    matches = sorted(root.rglob(filename))
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Expected one {filename} under {root}, found {len(matches)}.")


def read_json(root, filename, key):
    return json.loads(locate(root, filename).read_text())[key]


def choose_window(episodes, choice):
    if choice is None:
        for episode in episodes.values():
            print(episode["id"], episode["start"], episode["end"])
        choice = input("Event ID, or C for custom UTC times: ")
    choice = choice.strip().upper()
    if choice == "C":
        start = input("Start UTC (YYYY-MM-DD HH:MM:SS): ")
        end = input("End UTC (YYYY-MM-DD HH:MM:SS): ")
        return start, end, None
    if choice not in episodes:
        raise ValueError(f"Unknown episode {choice!r}.")
    episode = episodes[choice]
    return episode["start"], episode["end"], episode


def plot_window(telemetry, start, end, episode, feeds, reveal_label):
    fig, axes = plt.subplots(8, 1, sharex=True, figsize=(13, 12), layout="constrained")
    fields = [
        ("mean_cn0_dbhz", "Mean C/N₀ (dB-Hz)"),
        ("min_cn0_dbhz", "Min C/N₀ (dB-Hz)"),
        ("cross_track_error_m", "Cross-track error (m)"),
        ("num_sats", "Satellites"),
        ("fix_state", "Fix state"),
        ("base_link_ok", "Base link"),
    ]
    states = ["NO_FIX", "AUTONOMOUS", "DGPS", "RTK_FLOAT", "RTK_FIXED"]
    state_codes = {state: index for index, state in enumerate(states)}
    for ax, (field, label) in zip(axes, fields):
        ax.set_ylabel(label)
        for tractor, rows in telemetry.groupby("tractor_id"):
            values = rows[field]
            if field == "fix_state":
                ax.scatter(rows.ts, values.map(state_codes), s=4, label=tractor)
            elif field == "base_link_ok":
                ax.step(rows.ts, values.astype(int), where="post", label=tractor)
            else:
                ax.plot(rows.ts, values, linewidth=0.8, label=tractor)
    axes[0].legend(loc="upper right", ncol=3)
    axes[4].set_yticks(range(len(states)), states, fontsize=7)
    axes[5].set_yticks([0, 1], ["Down", "Up"])

    for ax, (records, key, label) in zip(axes[6:], feeds[:2]):
        series = pd.DataFrame(records)
        series["ts"] = pd.to_datetime(series["ts"], utc=True)
        series = series.sort_values("ts")
        ax.set_ylabel(label)
        if key == "kp":
            prior = series[series.ts < start].tail(1)
            inside = series[(series.ts >= start) & (series.ts <= end)]
            points = pd.concat([prior, inside])
            if not points.empty:
                ax.step([*points.ts, end], [*points.kp, points.kp.iloc[-1]], where="post")
        else:
            inside = series[(series.ts >= start) & (series.ts <= end)]
            ax.plot(inside.ts, inside[key], marker=".", linewidth=0.9)

    for ax in axes:
        if episode:
            ax.axvspan(pd.Timestamp(episode["start"]), pd.Timestamp(episode["end"]),
                       color="gold", alpha=0.18)
        ax.grid(axis="y", alpha=0.2)

    for records, color, kind in feeds[2:]:
        for record in records:
            timestamp = pd.to_datetime(record["ts"], utc=True)
            if start <= timestamp <= end:
                for ax in axes:
                    ax.axvline(timestamp, color=color, linestyle="--", alpha=0.65)
                print(f"{timestamp} | {kind} {record['id']} | {record['text']}")
            elif kind == "Advisory" and timestamp < start:
                print(f"{timestamp} | Earlier advisory {record['id']} | {record['text']}")

    title = episode["id"] if episode else "Custom window"
    if episode and reveal_label:
        title += f" | {episode['class']} | {', '.join(episode['tractors'])}"
    fig.suptitle(f"{title}   •   Orange shading = labeled window   •   Red = RFI   •   Purple = advisory")
    axes[-1].set_xlim(start, end)
    locator = mdates.AutoDateLocator()
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[-1].set_xlabel("Time (UTC)")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Dataset root (default: ./data, ./upload, or .)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--event", help="Labeled episode ID, for example TRN-03")
    selection.add_argument("--window", nargs=2, metavar=("START", "END"), help="Custom UTC times")
    parser.add_argument("--context", type=int, default=None, help="Extra minutes before and after")
    parser.add_argument("--reveal-label", action="store_true", help="Show the known cause and tractors")
    parser.add_argument("--save", type=Path, help="Save the chart to a file instead of opening a window")
    args = parser.parse_args()

    if args.data is None:
        args.data = next((path for path in (Path("data"), Path("upload")) if path.is_dir()), Path("."))
    try:
        episode = None
        if args.window:
            start, end = args.window
        else:
            records = read_json(args.data, "train_episodes.json", "episodes")
            episodes = {record["id"]: record for record in records}
            start, end, episode = choose_window(episodes, args.event)
        start = pd.to_datetime(start, utc=True)
        end = pd.to_datetime(end, utc=True)
        if pd.isna(start) or pd.isna(end) or end <= start:
            raise ValueError("Enter a valid start and a later end.")
        context = args.context if args.context is not None else (20 if episode else 0)
        if context < 0:
            raise ValueError("--context must be zero or positive.")
        start -= pd.Timedelta(minutes=context)
        end += pd.Timedelta(minutes=context)
        telemetry = pd.read_csv(locate(args.data, "tractor_telemetry.csv.gz"))
        telemetry["ts"] = pd.to_datetime(telemetry["ts"], utc=True)
        rows = telemetry[telemetry.ts.between(start, end)].sort_values("ts")
        if rows.empty:
            raise ValueError("No tractor telemetry in that UTC window.")
        feeds = [
            (read_json(args.data, "sgi_indices.json", "kp"), "kp", "Kp"),
            (read_json(args.data, "scintillation_series.json", "series"), "s4", "S4"),
            (read_json(args.data, "rfi_reports.json", "reports"), "firebrick", "RFI"),
            (read_json(args.data, "constellation_advisories.json", "advisories"), "purple", "Advisory"),
        ]
        figure = plot_window(rows, start, end, episode, feeds, args.reveal_label)
        if args.save:
            figure.savefig(args.save, dpi=150)
            print(f"Saved {args.save}")
        else:
            plt.show()
    except (ValueError, OSError, EOFError) as error:
        parser.exit(2, f"event_explorer: {error}\n")


if __name__ == "__main__":
    main()
