import json

import pandas as pd

# These settings control sensitivity.
# The actual C/N₀ and cross-track thresholds will be learned from data.
LOW_PERCENTILE = 0.01
HIGH_PERCENTILE = 0.99

# Only use the labeled development period to learn normal behavior.
# May 6 is the evaluation day.
EVALUATION_START = pd.Timestamp("2026-05-06T00:00:00Z")


# Load every tractor's measurements.
telemetry = pd.read_csv(
    "data/farm/tractor_telemetry.csv.gz",
    parse_dates=["ts"],
)

# Make sure timestamps are consistently interpreted as UTC.
telemetry["ts"] = pd.to_datetime(telemetry["ts"], utc=True)

# Keep rows ordered by tractor and time for later interval detection.
telemetry = telemetry.sort_values(["tractor_id", "ts"])


# Load the known training events.
with open("data/labels/train_episodes.json") as file:
    episodes = json.load(file)["episodes"]


# Begin with all readings before the evaluation day.
normal_reference = telemetry["ts"] < EVALUATION_START

for episode in episodes:
    start = pd.Timestamp(episode["start"])
    end = pd.Timestamp(episode["end"])

    # This is True only for rows belonging to an affected tractor.
    affected_tractor = telemetry["tractor_id"].isin(episode["tractors"])

    # Treat the episode end as exclusive.
    during_episode = (telemetry["ts"] >= start) & (telemetry["ts"] < end)

    # Remove known event readings from our normal reference.
    # ~ means NOT. & means AND.
    normal_reference = normal_reference & ~(affected_tractor & during_episode)

# Store each tractor's completed table here.
flagged_tractors = []

for tractor_id, rows in telemetry.groupby("tractor_id"):
    # Select this tractor's normal reference readings.
    normal_rows = rows.loc[normal_reference.loc[rows.index]]

    if normal_rows.empty:
        raise ValueError(f"No normal reference readings for {tractor_id}")

    # Learn a lower signal-strength limit for this tractor.
    cn0_lower_limit = normal_rows["mean_cn0_dbhz"].quantile(LOW_PERCENTILE)

    # Use distance magnitude so either direction of path error counts.
    error_upper_limit = (
        normal_rows["cross_track_error_m"].abs().quantile(HIGH_PERCENTILE)
    )

    # Make a separate table where we can add our calculated columns.
    flagged = rows.copy()

    # Keep the learned limits alongside the measurements.
    # This lets us inspect why a reading was flagged.
    flagged["cn0_lower_limit"] = cn0_lower_limit
    flagged["error_upper_limit"] = error_upper_limit

    # Each comparison creates one True/False value for every reading.
    flagged["low_cn0"] = flagged["mean_cn0_dbhz"] < cn0_lower_limit

    flagged["high_error"] = flagged["cross_track_error_m"].abs() > error_upper_limit

    # A reading is a candidate anomaly if either symptom is present.
    flagged["anomaly"] = flagged["low_cn0"] | flagged["high_error"]

    flagged_tractors.append(flagged)

    print(
        f"{tractor_id}: "
        f"C/N₀ lower limit = {cn0_lower_limit:.2f}, "
        f"path-error upper limit = {error_upper_limit:.3f} m"
    )


# Combine all tractors back into one table.
flagged_data = pd.concat(flagged_tractors).sort_values(["tractor_id", "ts"])

# Show a few flagged readings and the symptoms responsible.
columns_to_show = [
    "ts",
    "tractor_id",
    "mean_cn0_dbhz",
    "cross_track_error_m",
    "low_cn0",
    "high_error",
]

print(
    flagged_data.loc[flagged_data["anomaly"], columns_to_show]
    .head(20)
    .to_string(index=False)
)
