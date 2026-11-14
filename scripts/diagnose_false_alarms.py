"""Audit development predictions without treating May 6 as labeled.
Run: python scripts/diagnose_false_alarms.py --data data --results results.json
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import detect
from ui_core import read_results


def audit(root, result_path):
    data, labels, feeds = detect.load_data(root)
    result = read_results(result_path.read_text())
    flags = detect.create_flags(data,result['learned_limits'],result['settings'])
    development = flags.ts < detect.EVAL_START
    labeled = pd.Series(False,index=flags.index)
    for label in labels:
        labeled |= flags.tractor_id.isin(label['tractors']) & (flags.ts>=pd.Timestamp(label['start'])) & (flags.ts<pd.Timestamp(label['end']))
    # "Outside labels" is not a proof of falsehood: recurrent multipath can
    # exist outside the sparse example labels. Keep those distinctions explicit.
    background = development & ~labeled
    rates=[]
    for moving in (True,False):
        rows=flags[background & ((flags.speed_mps>0)==moving)]
        rates.append({'stratum':'moving' if moving else 'parked','rows':len(rows), **{f:round(float(rows[f].mean()),6) for f in detect.FLAGS}})
    tractor_rows={tractor:rows.sort_values("ts").set_index("ts",drop=False) for tractor,rows in flags.groupby("tractor_id")}
    members=[]
    for event_id,event in enumerate(result['events']):
        for member in event['members']:
            if member['start']>=detect.EVAL_START: continue
            tractor=member['tractors'][0]
            overlaps=[l for l in labels if tractor in l['tractors'] and member['start']<pd.Timestamp(l['end']) and member['end']>pd.Timestamp(l['start'])]
            status='matching_cause_overlap' if any(l['class']==member['class'] for l in overlaps) else 'other_cause_label_overlap' if overlaps else 'outside_all_labels'
            rows=tractor_rows[tractor]
            segment=rows.iloc[rows.index.searchsorted(member['start']):rows.index.searchsorted(member['end'])]
            members.append({'event':event_id,'tractor':tractor,'cause':member['class'],'trigger':member['trigger'],'status':status,'duration_s':(member['end']-member['start']).total_seconds(),'moving_fraction':float((segment.speed_mps>0).mean()),'start':str(member['start']),'end':str(member['end']),'winning_score':member['scores'][member['class']]})
    frame=pd.DataFrame(members)
    grouped=frame.groupby(['status','trigger','cause']).agg(candidates=('event','size'),median_seconds=('duration_s','median'),parked_candidates=('moving_fraction',lambda s:int((s==0).sum()))).reset_index()
    development_members=[m for event in result['events'] for m in event['members'] if m['start']<detect.EVAL_START]
    multipath=[m for m in development_members if m['class']=='MULTIPATH']
    short_spoofing=[m for m in development_members if m['class']=='SPOOFING' and (m['end']-m['start']).total_seconds()<=60]
    duplicated=sum(any(set(m['tractors']) & set(n['tractors']) and m['start']<n['end'] and m['end']>n['start'] for n in multipath) for m in short_spoofing)
    summary={'development_short_spoofing_candidates':len(short_spoofing),'development_short_spoofing_overlapping_multipath':duplicated,'note':'Development only. Outside labels is potential false alarm, not confirmed falsehood; sparse labels do not enumerate all recurrent multipath. May 6 excluded. Counts refer to individual candidates, not merged events.', 'background_flag_rates':rates,'candidate_status_counts':frame.status.value_counts().to_dict(),'by_trigger_and_cause':grouped.to_dict('records'),'examples_outside_labels':frame[frame.status=='outside_all_labels'].head(12).to_dict('records'),'learned_limits':result['learned_limits']}
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data'))
    p.add_argument('--results',type=Path,default=Path('results.json'))
    args=p.parse_args()
    print(json.dumps(audit(args.data,args.results),indent=2))
