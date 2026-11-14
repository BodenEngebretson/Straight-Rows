# Detector grouping update: operational-grouping-v2

This update changes actual CLI detections, rather than only filtering a table.
It follows the diagnosis in [FALSE_ALARM_ANALYSIS.md](FALSE_ALARM_ANALYSIS.md).
The development labels informed these choices; this is not independent evaluation.
No hidden labels or target event counts were used.

## What changed

- Candidate construction uses moving telemetry. Parked measurements remain
  browsable, but parked-only symptoms cannot create operational predictions.
- Same-cause fragments join across up to **180 seconds**, only if they share a
  tractor and all intervening one-second moving samples exist. A connected-component
  sweep handles interleaved tractors and transitive links. Different causes stay
  separate. Original member start/end, hits, densities, and cause scores survive.
- Short SPOOFING hypotheses from weak minimum C/N₀ or high error are suppressed
  if at least 80% overlaps same-tractor MULTIPATH, signal spread occupies at least
  50% of the candidate, and excess error occupies less than 50%.
- Other spoofing candidates shorter than 120 seconds need at least 50%
  fleet-relative excess-error support in a centered 300-second operational window,
  and no more than 5% link-failure samples there. This is a heuristic support gate,
  not proof that a brief attack cannot occur. It can reject real brief attacks.
- Jamming candidates at most 60 seconds long need severe signal collapse or
  matching RFI corroboration. Signal spread during broad mean-signal collapse
  alone no longer creates a separate equipment-fault alert.
- A second signal-spread construction detects intermittent antenna symptoms:
  recoveries up to 60 seconds, at least 30 hits, density at least 0.15, span at
  least 120 seconds, and low-mean-C/N₀ fraction below 0.5. This supplements the
  original short-multipath path. Accepted candidates use the existing classifier
  and existing long signal-spread equipment bonus; their exact settings are saved.

The classifier's numerical weights and learned measurement thresholds are
unchanged. Suppression happens after scoring. `pipeline_diagnostics` saves every
suppressed candidate, its original scores and a reason; no score is silently
changed to force a preferred cause. The new persistence/context settings appear
in `DEFAULTS`. Duplicate-resolution and corroboration constants are documented
above and in the detector source.

## Observed result on this dataset

| Output | Before | After |
|---|---:|---:|
| Operational alert groups (all causes except MULTIPATH) | 532 | 20 |
| MULTIPATH reports | 701 | 444 |
| Total groups/reports | 1,233 | 464 |

These totals are not capped. Repeated multipath remains visible in its own scope;
“All predictions” shows everything. In the new pipeline, 4,305 raw symptom
candidates become 2,202 retained candidates after 2,103 suppressions, then group
into 464 output intervals. The difference in raw candidates versus the old 4,327
also reflects excluding parked candidates and adding the antenna persistence path.

The previously missed TRN-09 equipment fault now appears as one T2 interval,
**May 4 11:59:51–13:59:59 UTC**, with 100% labeled member-time coverage after
rounding. Its overlapping spoofing interval remains a separate cause.
TRN-10's correction-link fragments now form one group, **May 5
02:00:01–02:47:57 UTC**, while their original member boundaries remain visible.
The recovered gaps are inside the group's envelope but are not drawn as flags.

The update still has errors. Three development spoofing groups occur during
jamming and need further evidence review. TRN-08's spoofing group begins late;
actual member coverage falls from 68.1% to 62.3%. TRN-10 member-time coverage is
68.9%, even though its grouped envelope closely matches the labeled episode.
The supplied labels are sparse for recurring multipath, and May 6 remains
unlabeled; its predictions cannot be claimed correct from this comparison.

## UI behavior

Both the development-episode table and prediction table support multiple row
selections. All selected predictions shade their actual member-support spans in
cause-specific colors. The visible window fits the earliest start through the
latest end with the chosen context. Gaps between grouped member intervals are
not shaded. Choose **Explain selected event** to change the detailed explanation
without clearing the other highlights. **Show only selected predictions on the
timing chart** reduces unrelated clutter.

The default prediction scope is **Operational alerts**. Use **Recurring multipath
reports** or **All predictions** to inspect the other reports. Summary counts
separate the two types. The completion message no longer shows a temporary file
path. New results carry `detector_version: operational-grouping-v2`; loading an
old file warns that its predictions have not been recomputed.

A current run is saved in `results_grouped.json`. Click **Load results file** to
inspect it immediately, or **Run Detector** to produce a fresh run.

## Checks

```bash
python detect.py --data data --output results_grouped.json --validate
python detect.py --self-test
python -m unittest test_grouping test_ui -v
```

Grouping tests cover interleaved tractors, transitive bridges, maximum recovery,
missing/parked samples, different causes, original member boundaries, and duplicate
resolution versus independent excess-error evidence. UI tests cover selecting two
labels and two predictions, simultaneous chart annotations, and changing the
explained event without clearing the selection. A UI-triggered run is compared
against an independent CLI run, including pipeline diagnostics and every score.

## Subsequent v3 verification and UI simplification

The stricter one-to-one check in `verification.py` found that the v2 results above
matched 10/12 labels: broad cross-tractor/recovery-gap multipath grouping failed
TRN-11 and TRN-12 despite covering their measurement flags. v3 keeps MULTIPATH
reports per tractor and merges only overlapping members for that cause. It now
matches 12/12 under the guide's cause, time-IoU and tractor-IoU thresholds. This
changes multipath report count from 444 to 1,760; operational groups remain 20.
These are separate recurring reports, not 1,760 additional scored scenario events.

The UI has also been simplified to interval selection and one linked chart stack.
Detailed event panels and the timing/flag charts described earlier were removed.
See [the current UI guide](../UI_GUIDE.md) for current controls and verification.
