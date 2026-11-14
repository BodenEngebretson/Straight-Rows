"""Launch: python -m streamlit run app.py"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import detect
from ui_core import read_results, utc_window, result_table, envelope

from verification import development_report

st.set_page_config(page_title='Straight Rows', layout='wide')
st.title('Straight Rows')

@st.cache_data(show_spinner='Reading telemetry and external observations…')
def dataset(path, signature):
    return detect.load_data(Path(path))

with st.sidebar:
    st.header('Data & execution')
    path = st.text_input('Dataset directory', 'data')
    load = st.button('Load dataset', type='primary')
    st.caption('Runs use the entire loaded dataset, including development reference data. Browsing never runs detection.')

if load or 'data' not in st.session_state:
    try:
        root = Path(path).resolve()
        signature = tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in sorted(root.rglob('*')) if p.is_file())
        data, labels, feeds = dataset(str(root), signature)
        st.session_state.update(data=data, labels=labels, feeds=feeds, root=str(root))
        st.session_state.pop('result', None)
        st.session_state.pop('run_summary', None)
    except Exception as exc:
        st.error(f'Dataset could not be loaded: {exc}')
        if 'data' not in st.session_state:
            st.stop()
data, labels, feeds = (st.session_state[k] for k in ('data','labels','feeds'))
root = st.session_state.root
st.caption(f"{Path(root).name} · {data.ts.min()} — {data.ts.max()} · UTC")
with st.sidebar:
    run = st.button('Run Detector', type='primary', key='run_detector')
    upload = st.file_uploader('Load diagnostic results JSON', type=['json'])
    result_path = st.text_input('Or local results path', 'results_grouped.json', key='results_path')
    load_result = st.button('Load results file', key='load_results')
    st.caption('Load results only from this dataset. Older results have no dataset fingerprint; verify their source.')
if run:
    started = time.monotonic()
    with st.sidebar, st.status('Running detector…', expanded=True) as status:
        progress = st.empty()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)/'results.json'
                process = subprocess.Popen([sys.executable, str(Path(__file__).with_name('detect.py')), '--data', root, '--output', str(output)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                while process.poll() is None:
                    progress.write(f'Running · {time.monotonic()-started:.1f} seconds elapsed')
                    time.sleep(.25)
                stdout, stderr = process.communicate()
                if process.returncode:
                    raise RuntimeError(stderr or stdout)
                result = read_results(output.read_text())
                result['dataset'] = root
                st.session_state.result = result
                operational_count = sum(e['class'] != 'MULTIPATH' for e in result['events'])
                recurring_count = len(result['events']) - operational_count
                st.session_state.run_summary = f"Completed in {time.monotonic()-started:.1f}s · {operational_count} operational alert groups · {recurring_count} multipath reports"
                status.update(label=st.session_state.run_summary, state='complete')
        except Exception as exc:
            status.update(label=f'Failed after {time.monotonic()-started:.1f}s', state='error')
            st.error(str(exc))
if load_result or (upload is not None and st.session_state.get('uploaded_id') != upload.file_id):
    try:
        result = read_results(upload.getvalue().decode() if upload else Path(result_path).read_text())
        st.session_state.result = result
        st.session_state.run_summary = f"Loaded {len(result['events'])} predicted intervals from existing results; no detector run."
        if upload:
            st.session_state.uploaded_id = upload.file_id
    except Exception as exc:
        st.error(f'Results file could not be loaded: {exc}')
result = st.session_state.get('result')

with st.sidebar:
    if 'run_summary' in st.session_state:
        st.caption(st.session_state.run_summary)
    if result and result.get('detector_version') != 'operational-grouping-v3':
        st.warning('Old results loaded. Run Detector for current grouping.')
    st.divider()
    tractors = st.multiselect('Tractors', sorted(data.tractor_id.unique()), default=sorted(data.tractor_id.unique()), key='tractors')
    context_before = st.number_input('Minutes before interval', 0, 1440, 10, key='context_before')
    context_after = st.number_input('Minutes after interval', 0, 1440, 10, key='context_after')
    focus = st.toggle('Fit selected intervals', True, key='fit_selection')
    show_labels = st.toggle('Overlay development labels', False, key='overlay_labels')

report = development_report(result['events'], labels) if result else []
if result:
    matches = sum(row['Status'] == 'Matched' for row in report)
    with st.expander(f'Development check: {matches}/{len(report)} episodes matched', expanded=False):
        st.dataframe(pd.DataFrame(report),hide_index=True,width='stretch',alt='One-to-one detector matches for development labels')
        st.caption('Same cause, time IoU ≥ 0.30, tractor IoU ≥ 0.50; one prediction per label. A match does not imply exact boundaries. May 6 is unlabeled.')

view = st.radio('Intervals', ['Known development episodes', 'Detector predictions'], horizontal=True, key='browse_mode', label_visibility='collapsed')
chosen_labels = []
selected_ids = []
if view == 'Known development episodes':
    status_by_id = {row['Episode']:row for row in report}
    filtered_labels = [l for l in labels if set(l['tractors']) & set(tractors)]
    table = pd.DataFrame([{'Episode':l['id'],'Start UTC':l['start'],'End UTC':l['end'],'Cause':l['class'],'Tractors':', '.join(l['tractors']),'Detector':status_by_id.get(l['id'],{}).get('Status','Not run')} for l in filtered_labels],columns=['Episode','Start UTC','End UTC','Cause','Tractors','Detector'])
    picked = st.dataframe(table,hide_index=True,height=240,width='stretch',on_select='rerun',selection_mode='multi-row',key='known_episodes_table',alt='Select development intervals to highlight')
    chosen_labels = [filtered_labels[i] for i in picked.selection.rows if 0 <= i < len(filtered_labels)]
    selected_ids = [int(status_by_id[l['id']]['Prediction']) for l in chosen_labels if status_by_id.get(l['id'],{}).get('Status') == 'Matched']
else:
    if result:
        with st.sidebar:
            scope = st.selectbox('Prediction scope',['Operational alerts','Recurring multipath reports','All predictions'],key='prediction_scope')
        table = result_table(result['events'])
        table = table[table.ID.isin([i for i,e in enumerate(result['events']) if set(e['tractors']) & set(tractors) and (scope == 'All predictions' or (scope == 'Operational alerts' and e['class'] != 'MULTIPATH') or (scope == 'Recurring multipath reports' and e['class'] == 'MULTIPATH'))])]
        picked = st.dataframe(table[['ID','Start','End','Tractors','Cause']],hide_index=True,height=240,width='stretch',on_select='rerun',selection_mode='multi-row',key='events_table',alt='Select predicted intervals to highlight')
        selected_ids = [int(table.iloc[i].ID) for i in picked.selection.rows if 0 <= i < len(table)]
        if table.empty:
            st.info('No events match these filters.')
    else:
        st.info('Run the detector or load a results file to select predictions.')
selected_events = [result['events'][i] for i in selected_ids] if result else []
st.caption('Select multiple rows to compare intervals. Predictions are colored by cause; labels are orange. All times UTC.')

window_start,window_end = data.ts.min(),data.ts.min()+pd.Timedelta(hours=3)
if focus and (selected_events or chosen_labels):
    intervals = selected_events + chosen_labels
    window_start = min(pd.Timestamp(e['start']) for e in intervals)-pd.Timedelta(minutes=context_before)
    window_end = max(pd.Timestamp(e['end']) for e in intervals)+pd.Timedelta(minutes=context_after)
nav = f'{window_start}-{window_end}'
with st.sidebar:
    start_text = st.text_input('Visible start UTC',str(window_start),key='start'+nav)
    end_text = st.text_input('Visible end UTC',str(window_end),key='end'+nav)
try:
    start,end = utc_window(start_text,end_text)
except Exception as exc:
    st.error(str(exc)); st.stop()
rows = data[data.ts.between(start,end) & data.tractor_id.isin(tractors)].copy()
if rows.empty:
    st.warning('No measurements in this UTC window / tractor selection.')

colors = dict(zip(sorted(data.tractor_id.unique()), ['#007fba','#df6c14','#8e54bd']))
states = ['NO_FIX','AUTONOMOUS','DGPS','RTK_FLOAT','RTK_FIXED']
rows['fix_code'] = rows.fix_state.map(dict(zip(states,range(5))))
fields = [('mean_cn0_dbhz','Mean C/N₀ · dB-Hz'),('min_cn0_dbhz','Minimum C/N₀ · dB-Hz'),('cross_track_error_m','Cross-track error · m'),('num_sats','Satellites · count'),('fix_code','Fix state'),('base_link_ok','Base link'),('speed_mps','Speed · m/s'),('kp','Kp · index'),('s4','S4 · index')]
fig = make_subplots(rows=9, cols=1, shared_xaxes=True, vertical_spacing=.045)
for row,(field,title) in enumerate(fields,1):
    if row<=7:
        for tractor, group in rows.groupby('tractor_id'):
            points = envelope(group,field)
            fig.add_trace(go.Scattergl(x=points.ts,y=points[field].astype(int) if field=='base_link_ok' else points[field],name=tractor,legendgroup=tractor,showlegend=row==1,line=dict(color=colors[tractor],width=1.2,shape='hv' if field in ('fix_code','base_link_ok') else 'linear'),hovertemplate='%{x|%Y-%m-%d %H:%M:%S} UTC<br>%{y}<extra>'+tractor+'</extra>'), row=row,col=1)
    else:
        feed = pd.DataFrame(feeds[field]); feed.ts = pd.to_datetime(feed.ts,utc=True); feed=feed.sort_values('ts')
        points = pd.concat([feed[feed.ts<start].tail(1),feed[feed.ts.between(start,end)]])
        fig.add_trace(go.Scatter(x=points.ts,y=points[field],name=field.upper(),line=dict(shape='hv'),showlegend=False),row=row,col=1)
    fig.update_yaxes(title_text=title, title_standoff=18, automargin=True, row=row,col=1)
fig.update_yaxes(tickvals=list(range(5)),ticktext=states,tickfont=dict(size=10),range=[-.5,4.5],row=5,col=1)
fig.update_yaxes(tickvals=[0,1],ticktext=['Down','Up'],row=6,col=1)
cause_colors = {'SPACE_WEATHER':'#2274a5','JAMMING':'#bf4040','SPOOFING':'#8e54bd','EQUIPMENT_FAULT':'#24874b','MULTIPATH':'#6b7785'}
for event_id, event in zip(selected_ids, selected_events):
    # Shade member support, not recovered gaps inside a group.
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for member in sorted(event['members'], key=lambda m: m['start']):
        if not set(member['tractors']) & set(tractors): continue
        if spans and member['start'] <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], member['end']))
        else:
            spans.append((member['start'],member['end']))
    for a,b in spans:
        fig.add_vrect(x0=a,x1=b,fillcolor=cause_colors[event['class']],opacity=.14,line_width=0,row='all',col=1)
    fig.add_annotation(x=event['start'],y=1,yref='paper',text=f"#{event_id} {event['class']}",showarrow=False,font=dict(size=10,color=cause_colors[event['class']]),yanchor='bottom')
if show_labels or chosen_labels:
    visible_labels = labels if show_labels else chosen_labels
    for label in visible_labels:
        if not set(label['tractors']) & set(tractors) or pd.Timestamp(label['start']) >= end or pd.Timestamp(label['end']) <= start:
            continue
        fig.add_vrect(x0=label['start'],x1=label['end'],fillcolor='orange',opacity=.18,line_width=0,row='all',col=1)
        if label in chosen_labels:
            fig.add_annotation(x=label['start'],y=1,yref='paper',text=label['id'],showarrow=False,font=dict(color='#ad6700',size=11),yanchor='bottom')
observations=[]
for kind,color in [('rfi','#b52d40'),('advisories','#8051aa')]:
    for record in feeds[kind]:
        t=pd.Timestamp(record['ts'])
        if start<=t<=end:
            fig.add_vline(x=t,line_color=color,line_dash='dot',row='all',col=1)
            observations.append({'Kind':kind,**record})
            fig.add_trace(go.Scatter(x=[t],y=[0],mode='markers',marker=dict(color=color,symbol='diamond',size=10),name=kind,showlegend=False,text=[record.get('id','')+' · '+record.get('text','')],hovertemplate='%{x|%Y-%m-%d %H:%M:%S} UTC<br>%{text}<extra>'+kind+'</extra>'),row=8,col=1)
fig.update_layout(height=1800,margin=dict(l=165,r=35,t=65,b=55),template='plotly_white',hovermode='x unified',dragmode='zoom',legend=dict(orientation='h',y=1.035,x=0),uirevision=nav+start_text+end_text)
fig.update_xaxes(range=[start,end],showspikes=True,spikemode='across',tickformat='%m-%d %H:%M',title_text='UTC',row=9,col=1)
st.plotly_chart(fig,width='stretch',config={'scrollZoom':True,'displaylogo':False})
st.caption('Drag to zoom; toolbar to pan; double-click to reset. Red markers: RFI reports. Purple: advisories.')
