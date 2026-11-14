"""Pure UI helpers. Detection always uses the complete reference dataset."""
import json
import numpy as np
import pandas as pd
import detect


def read_results(text):
    result = json.loads(text)
    if not isinstance(result, dict) or not all(k in result for k in ('events', 'settings', 'learned_limits')):
        raise ValueError('Load detect.py diagnostic results, not submission.json.')
    if not isinstance(result['events'], list):
        raise ValueError('events must be a list')
    if (set(detect.DEFAULTS) - {'merge_gap_seconds','antenna_gap_seconds','antenna_minimum_hits','antenna_minimum_density','antenna_minimum_seconds','spoof_context_seconds','spoof_minimum_context_fraction'}) - result['settings'].keys():
        raise ValueError('Missing detector settings')
    if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in result['settings'].values()):
        raise ValueError('Detector settings must be finite numbers')
    result['settings'].setdefault('merge_gap_seconds', 0)  # Legacy files merged overlaps only.
    config = result['settings']
    if config['gap_seconds'] < 0 or config['minimum_hits'] < 1 or not 0 < config['minimum_density'] <= 1:
        raise ValueError('Invalid interval settings')
    required_limits = {'mean_cn0_dbhz', 'min_cn0_dbhz', 'cross_track_error_m', 'num_sats', 'severe_cn0', 'signal_spread'}
    for limits in result['learned_limits'].values():
        if required_limits - limits.keys() or not all(isinstance(v, (int,float)) and np.isfinite(v) for v in limits.values()):
            raise ValueError('Missing or invalid learned thresholds')
    for event in result['events']:
        if event['class'] not in detect.ACTIONS or not event['members']:
            raise ValueError('Invalid cause or missing individual members')
        for interval in [event, *event['members']]:
            interval['start'], interval['end'] = utc_window(interval['start'], interval['end'])
            if not isinstance(interval['tractors'], list) or not interval['tractors']:
                raise ValueError('Each interval needs affected tractors')
        for member in event['members']:
            if member['trigger'] not in detect.FLAGS or set(member['scores']) != set(detect.ACTIONS):
                raise ValueError('Missing symptom or scores')
            if not all(isinstance(v,(int,float)) and np.isfinite(v) for v in member['scores'].values()):
                raise ValueError('Invalid cause scores')
            if not isinstance(member['evidence'],list) or member['flagged_seconds'] < 1 or not 0 < member['density'] <= 1:
                raise ValueError('Invalid candidate evidence, hit count or density')
    return result


def utc_window(start, end):
    start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
    if pd.isna(start) or pd.isna(end) or end <= start:
        raise ValueError('Enter valid UTC times with end after start.')
    return start, end


def result_table(events):
    return pd.DataFrame([dict(ID=i, Start=str(e['start']), End=str(e['end']),
        Seconds=(pd.Timestamp(e['end'])-pd.Timestamp(e['start'])).total_seconds(),
        Tractors=', '.join(e['tractors']), Symptoms=', '.join(sorted({m['trigger'] for m in e['members']})),
        Cause=e['class'], Score=max(m['scores'][e['class']] for m in e['members']))
        for i,e in enumerate(events)], columns=['ID','Start','End','Seconds','Tractors','Symptoms','Cause','Score'])


def envelope(rows, field, budget=1400):
    """Keep each bucket's first, last, minimum and maximum in time order.

    Applied separately per measurement: single-sample spikes survive. Categorical
    fields retain every transition, even when that exceeds the drawing budget.
    """
    if len(rows) <= budget:
        return rows
    values = rows[field]
    if field in ('fix_code', 'base_link_ok'):
        transition = values.ne(values.shift()) | values.ne(values.shift(-1))
        return rows.loc[transition]
    buckets = np.arange(len(rows)) // max(1, int(np.ceil(len(rows)/(budget/4))))
    grouped = rows.groupby(buckets)[field]
    indices = set(grouped.idxmin()) | set(grouped.idxmax())
    indices |= set(rows.groupby(buckets).head(1).index) | set(rows.groupby(buckets).tail(1).index)
    return rows.loc[sorted(indices)].sort_values('ts')


def flag_runs(rows, flag):
    """Exact true spans for a flag raster; missing samples break spans."""
    hits = rows.loc[rows[flag], 'ts']
    if hits.empty:
        return []
    groups = hits.diff().ne(pd.Timedelta(seconds=1)).cumsum()
    return [(g.iloc[0], g.iloc[-1]+detect.SECOND) for _,g in hits.groupby(groups)]


def score_terms(fractions, trigger, seconds, s4, kp, rfi_confidence, config):
    """Readable arithmetic from classify(); checked against stored scores in UI."""
    f=fractions
    low,high,link=f['low_cn0'],f['high_error'],f['link_failure']
    elevated_s4 = bool(np.isfinite(s4) and s4>=config['s4_elevated'])
    elevated_kp = bool(np.isfinite(kp) and kp>=config['kp_elevated'])
    short=seconds<=config['short_event_seconds']
    terms={
        'SPACE_WEATHER': {'4 × elevated S4':4*elevated_s4,'elevated Kp':int(elevated_kp),'low signal fraction':low},
        'JAMMING': {'3 × low signal fraction':3*low,'3 × matched RFI confidence':3*rfi_confidence,'few satellites fraction':f['few_sats'],'S4 not elevated':int(not elevated_s4)},
        'SPOOFING': {'3 × high error fraction':3*high,'2 × (1 − low signal fraction)':2*(1-low),'−2 × link failure fraction':-2*link},
        'EQUIPMENT_FAULT': {'6 × link failure fraction':6*link,'2 × weak minimum fraction × (1 − low signal fraction)':2*f['weak_min_cn0']*(1-low)},
        'MULTIPATH': {'short interval: 4 × high error + 2 × degraded fix + (1 − link failure)':4*high+2*f['fix_degraded']+1-link if short else 0.},
    }
    bonuses={'link_failure':('EQUIPMENT_FAULT',3),'high_error':('SPOOFING',1),'severe_cn0':('JAMMING',4),'excess_error':('SPOOFING',4),'signal_spread':('MULTIPATH',6) if short else ('EQUIPMENT_FAULT',7)}
    if trigger in bonuses:
        cause,value=bonuses[trigger]; terms[cause]['trigger bonus: '+trigger]=value
    return terms


def union_seconds(spans):
    total=0.; previous=None
    for a,b in sorted(spans):
        if previous is None or a>=previous: total+=(b-a).total_seconds()
        elif b>previous: total+=(b-previous).total_seconds()
        previous=max(previous,b) if previous is not None else b
    return total


def label_comparison(events, labels, tractors):
    rows=[]
    for label in labels:
        start,end=utc_window(label['start'],label['end'])
        for tractor in set(label['tractors']) & set(tractors):
            members=[m for e in events if e['class']==label['class'] for m in e['members'] if tractor in m['tractors'] and m['start']<end and m['end']>start]
            covered=union_seconds([(max(start,m['start']),min(end,m['end'])) for m in members])
            rows.append({'Label':label['id'],'Tractor':tractor,'Cause':label['class'],'Start UTC':str(start),'End UTC':str(end),'Coverage %':round(100*covered/(end-start).total_seconds(),1),'First boundary offset s':(min(m['start'] for m in members)-start).total_seconds() if members else None,'Last boundary offset s':(max(m['end'] for m in members)-end).total_seconds() if members else None})
    return pd.DataFrame(rows)
