# Why the detector returns so many predictions

The main problem is cause attribution and duplicate predictions over recurring
short bursts, with additional parked-machine detections and interval fragmentation.
It is not simply hundreds of thousands of independent random measurement spikes.

This audit uses the current `results.json`, current saved limits/settings, and the
12 development labels. May 6 is excluded from label-based conclusions. Reproduce:

```bash
python scripts/diagnose_false_alarms.py --data data --results results.json > docs/false_alarm_audit.json
```

The machine-readable output is [false_alarm_audit.json](false_alarm_audit.json).
An overlapping label does not prove that a candidate is correct, and a candidate
outside labels does not prove that it is false. In particular, development labels
contain representative multipath episodes, not all recurrent multipath occurrences.

## 1. Recurring multipath is being reported again as spoofing

Across all three days, the saved output has 701 merged MULTIPATH predictions and
486 SPOOFING predictions. 304 spoofing predictions are exactly 17 seconds long.
The event guide describes recurring headland multipath (page 2). Page 4 explicitly
exempts recurrent multipath reports lasting at most three minutes from extra-event
penalties, except overlapping duplicates in the representative scored window.
Thus the 12 + 8 benchmark episodes are **not a maximum count of real disturbances**.
My earlier characterization of essentially all short intervals as false alarms
was too broad.

In development alone, **1,300 of 1,350 short spoofing candidates (96.3%) overlap a
MULTIPATH candidate on the same tractor**. These counts are individual symptom
candidates, not merged events. T1's signal-spread candidates before noon May 4
mostly recur every 355–356 seconds, consistent with repeated headland encounters.
That pattern supports the multipath explanation, though timing alone is not proof.

The mechanism is explicit in `classify()`:

- A short `signal_spread` candidate receives a +6 MULTIPATH bonus.
- The overlapping `high_error` candidate instead receives +1 SPOOFING.
- A short `weak_min_cn0` candidate receives neither bonus.
- Each candidate is scored independently; `merge_candidates()` merges only
  predictions of the same cause. It retains the incompatible classifications as
  separate output events, although they often arise from the same measurements.

Concrete example, T3, May 4 00:02:49–00:03:04 UTC: the `high_error` candidate has
high-error fraction 2/3, low-mean-signal fraction 0, no link failures and no fix
changes. SPOOFING = 3×(2/3) + 2×1 + 1 trigger bonus = **5.000**.
MULTIPATH = 4×(2/3) + 0 + 1 = **3.667**. Yet signal spread is present throughout
that candidate, and a neighboring signal-spread candidate chooses MULTIPATH.
The same burst becomes two causes because the scoring depends strongly on the
symptom that proposed the interval.

Real causes can overlap. Therefore globally deleting every overlapping cause
would also be wrong; compatible hypotheses need to be resolved from their
measurement evidence while preserving evidence of genuinely independent causes.

## 2. Parked reception is promoted to operational events

`create_flags()` applies `speed_mps > 0` only to `high_error` (and indirectly
`excess_error`). Low signal, weak minimum, degraded fix and signal spread are
allowed to construct events while parked.

Outside labeled development episodes there are 172,800 parked tractor-seconds.
Low signal and weak minimum flags occur in **9.375%**, degraded fix in **8.726%**.
Those flags generate 21 parked space-weather symptom candidates outside all
labels (15 fix-degraded, 3 low-signal, 3 weak-minimum), plus two parked multipath
candidates. They merge into fewer events, so these are not 23 unique incidents.
The README states that poor reception while parked alone is not a scored guidance
event. A movement/operational-impact gate is missing.

## 3. Single-symptom acceptance and forced cause assignment

In moving development rows outside all supplied labels:

| Symptom | Fraction of rows flagged |
|---|---:|
| High cross-track error | 3.1464% |
| Weak minimum C/N₀ | 3.7981% |
| Signal spread | 4.5059% |
| Low mean C/N₀ | 0.2334% |

These background rows include recurring multipath. They are not a clean quiet-data
sample. The learned absolute-error limit is only **0.067 m** for all tractors;
normal headland disturbances can exceed it. The rule accepts eight hits with up
to three intervening unflagged seconds, density at least 0.35. Each symptom alone
can propose a candidate, even without independent guidance degradation.

`classify()` always picks an argmax. It has no unknown/insufficient-evidence
outcome, minimum absolute support, or minimum margin over the runner-up.
SPOOFING receives two points just for absence of low mean C/N₀; JAMMING receives
one point whenever S4 is not elevated, including missing S4. Neither condition
is affirmative evidence of an attack. This makes weak candidates look definite.
The heuristic numbers are not calibrated probabilities.

## 4. Fragmentation and incomplete merging

Each tractor and symptom has independent interval boundaries. Most allow only
three recovered seconds. Intermittent flags repeatedly stop and restart, and
same-cause predictions only merge when they strictly overlap. Touching or nearby
fragments remain separate. Link failure has a longer 60-second recovery allowance,
but other symptoms of intermittent antenna faults do not. This also helps explain
why the two-hour TRN-09 antenna-fault label is not recovered as one equipment event.

Among development candidates, 1,923 lie outside every supplied label, 763 overlap
only other-cause labels, and 216 overlap a matching-cause label. These figures
locate review targets; they are **not precision/recall** or proven false-alarm counts.
The 1,923 outside-label candidates include 656 recurring MULTIPATH candidates and
1,234 SPOOFING candidates triggered by high error or weak minimum C/N₀.

## Recommended next detector changes

1. Gate operational alerts on movement or demonstrable operational impact;
   retain parked symptoms as diagnostics rather than guidance incidents.
2. Compare competing hypotheses on a common local measurement window. Require
   sustained independent spoofing evidence before elevating a short multipath-like
   burst into a second attack alert. Preserve genuinely overlapping causes.
3. Add an explicit insufficient-evidence outcome and support/margin checks.
   Remove the inference that low/missing S4 itself supports jamming.
4. Use symptom-appropriate persistence and recovery rules, especially for
   intermittent equipment symptoms. Preserve original member boundaries and
   evaluate onset, duration, affected tractors, and duplicate predictions.
5. Check every change against all 12 labels, including the 17-second multipath
   examples and overlapping causes. Do not tune against hidden May 6 answers or
   force exactly eight evaluation predictions.

No detector weights, thresholds, or outputs were changed during this investigation.
The UI layout change increases panel gaps, total height, and label margins.
