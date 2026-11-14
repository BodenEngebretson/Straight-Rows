import matplotlib.pyplot as plt
import pandas as pd

telemetry = pd.read_csv("data/farm/tractor_telemetry.csv.gz", parse_dates=["ts"])

signals = ["mean_cn0_dbhz", "cross_track_error_m"]
minute = (
    telemetry.set_index("ts").groupby("tractor_id")[signals].resample("1min").mean()
)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 7))

for signal, label, ax in zip(signals, ["C/N₀ (dB-Hz)", "Cross-track error (m)"], axes):
    minute[signal].unstack("tractor_id").plot(ax=ax)
    ax.set_ylabel(label)
    ax.axvline(pd.Timestamp("2026-05-06T00:00:00Z"), color="black", linestyle="--")

axes[-1].set_xlabel("Time (UTC)")
fig.tight_layout()
axes[0].set_xlim(
    pd.Timestamp("2026-05-04 16:10:00Z"),
    pd.Timestamp("2026-05-04 17:32:00Z"),
)
fig.savefig("telemetry.png", dpi=150)
