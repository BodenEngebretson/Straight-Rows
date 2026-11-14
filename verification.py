"""One-to-one development checks using the guide's matching thresholds.

Labels are comparison data only, never detection inputs here. A match requires
same cause, temporal IoU >= .30 and affected-tractor IoU >= .50. Maximize match
count first, then total temporal overlap (seconds). May 6 is never labeled.
"""
import pandas as pd
import detect


def development_report(events, labels):
    labels = [l for l in labels if pd.Timestamp(l['end']) <= detect.EVAL_START]
    edges = {}
    for event_id, event in enumerate(events):
        start, end = pd.Timestamp(event['start']), pd.Timestamp(event['end'])
        if start >= detect.EVAL_START:
            continue
        end = min(end, detect.EVAL_START)
        for label_id, label in enumerate(labels):
            if event['class'] != label['class']:
                continue
            a,b = pd.Timestamp(label['start']), pd.Timestamp(label['end'])
            overlap = max(0., (min(end,b)-max(start,a)).total_seconds())
            union = (max(end,b)-min(start,a)).total_seconds()
            tiou = overlap/union
            predicted = {t for m in event['members'] if pd.Timestamp(m['start']) < detect.EVAL_START for t in m['tractors']}
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
