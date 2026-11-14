# Straight Rows local charts

Install and launch from the repository root (Python 3.11+):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

Open http://localhost:8501. Windows activation is `.venv\Scripts\Activate.ps1`.

The page contains an interval table and one linked measurement-chart stack.
Select multiple table rows to highlight them together. Selecting labeled episodes
also highlights their matching predictions when results are loaded. Use the sidebar
to choose tractors, context minutes, visible UTC times, or label overlays.
Drag charts to zoom, use the toolbar to pan, and double-click to reset.
Prediction shading follows actual member-support spans, not recovered grouping gaps.

Click **Run Detector** for a fresh full-dataset run, or **Load results file** to
load the current `results_grouped.json`. Execution is separate from browsing.
The sidebar shows execution status/errors. Event explanations, score tables,
raw records, source code, flag raster, and the extra timing chart have been removed.
The detector's diagnostic results still contain the underlying evidence.

## Checking TRN-01 through TRN-12

Every completed CLI run saves a `development_verification` report. The UI also
computes the check when a results file is loaded, so old results are checked too.
A small collapsed **Development check** above the interval selector shows the
matched count; expand it for each label's result and boundary offsets.

The matching criteria from the event guide are:

- Same predicted cause.
- Time intersection-over-union (IoU) at least 0.30.
- Affected-tractor set IoU at least 0.50.
- One-to-one assignment, maximizing match count first and total overlapping time
  second. One broad prediction cannot count as several correctly detected episodes.

The current v3 output matches **12/12** development episodes. Matching does not
mean perfect boundaries or no false positives: TRN-08 starts 767 seconds late,
TRN-02 ends 360 seconds late, and TRN-04 includes an extra tractor.
Development labels also influenced threshold training and rule development;
this is not independent evaluation. May 6 is unlabeled and excluded from checks.

The new matching check found that v2's broad MULTIPATH grouping failed TRN-11
and TRN-12, despite 100% member coverage. v3 keeps multipath per tractor and joins
only overlapping members, preserving short episode boundaries. Operational alert
grouping is unchanged. The current file has 20 operational groups and 1,760 brief
per-tractor multipath reports. The default prediction view shows operational alerts;
choose another prediction scope to inspect multipath. Counts are not capped.

## Code and checks

`app.py` calls the existing detector CLI against the complete dataset.
`detect.py` constructs candidates and groups them; `ui_core.py` handles plotting
reduction, result loading and tables. `verification.py` performs label matching
without changing detections. Numeric plots keep per-bucket extrema so brief spikes
survive; categorical plots keep transitions. All timestamps are UTC and interval
ends are exclusive.

```bash
python detect.py --data data --output results_grouped.json --validate
python detect.py --self-test
python -m unittest test_grouping test_verification test_ui -v
```

Tests cover one-to-one matching, cause/tractor checks, tiny overlaps, evaluation
exclusion, brief per-tractor multipath, multiple highlights, absent files, empty
results, UTC filtering, spike retention, and exact UI/CLI run equality.

See [the false-alarm diagnosis](docs/FALSE_ALARM_ANALYSIS.md) and
[the grouping update](docs/GROUPING_UPDATE.md) for the preceding investigation
and detector-rule rationale. Their v2 output counts describe the earlier version.
