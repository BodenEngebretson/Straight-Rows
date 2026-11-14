import json
import unittest
from pathlib import Path
import pandas as pd
from streamlit.testing.v1 import AppTest
from ui_core import read_results, utc_window, envelope, result_table, score_terms

class UI(unittest.TestCase):
    def test_helpers(self):
        a,b=utc_window('2026-05-04T01:00:00+01:00','2026-05-04T00:01:00Z')
        self.assertEqual(a.hour,0)
        with self.assertRaises(ValueError): utc_window(b,a)
        for bad in ['{}','bad','{"events":[]}']:
            with self.assertRaises((ValueError,KeyError)): read_results(bad)
        r=read_results(json.dumps({'events':[], 'settings':__import__('detect').DEFAULTS,'learned_limits':{}}))
        self.assertTrue(result_table(r['events']).empty)
        rows=pd.DataFrame({'ts':pd.date_range('2026-05-04',periods=10000,freq='s',tz='UTC'),'v':0.})
        rows.loc[457,'v']=99
        self.assertEqual(envelope(rows,'v').v.max(),99)

    def test_app(self):
        app=AppTest.from_file('app.py',default_timeout=120).run()
        self.assertFalse(app.exception)
        app.session_state['known_episodes_table']={'selection':{'rows':[2,3],'columns':[],'cells':[]}}
        app.run()
        self.assertFalse(app.exception)
        spec=json.loads(app.get('plotly_chart')[0].proto.spec)
        self.assertEqual(sum(a.get('text','').startswith('TRN-') for a in spec['layout']['annotations']),2)
        app.radio[0].set_value('Detector predictions').run()
        app.button(key='load_results').click().run()
        self.assertFalse(app.exception)
        self.assertIn('result',app.session_state)
        self.assertTrue(any('12/12 episodes matched' in e.label for e in app.expander))
        app.session_state['events_table']={'selection':{'rows':[0,1],'columns':[],'cells':[]}}
        app.run()
        self.assertFalse(app.exception)
        spec=json.loads(app.get('plotly_chart')[0].proto.spec)
        annotations=[a for a in spec['layout'].get('annotations',[]) if a.get('text','').startswith('#')]
        self.assertEqual(len(annotations),2)
        self.assertEqual(len(app.get('plotly_chart')),1)
        self.assertFalse(app.code)
        self.assertFalse(app.json)
        self.assertFalse(any(x.label in ('Explain selected event','Inspect individual candidate') for x in app.selectbox))
        next(x for x in app.text_input if x.label=='Or local results path').set_value('/tmp/absent-straight-rows.json')
        app.button(key='load_results').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertIn('result',app.session_state)
        next(x for x in app.text_input if x.label=='Dataset directory').set_value('/tmp/absent-straight-rows-data')
        next(b for b in app.button if b.label=='Load dataset').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertIn('result',app.session_state)
        empty=dict(app.session_state['result']); empty['events']=[]
        app.session_state['result']=empty
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any('No events match' in x.value for x in app.info))

    def test_run_parity(self):
        app=AppTest.from_file('app.py',default_timeout=120).run()
        next(b for b in app.button if b.label=='Run Detector').click().run()
        self.assertFalse(app.exception)
        actual=json.loads(json.dumps(app.session_state['result'],default=str))
        actual.pop('dataset',None)
        import subprocess, sys, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'cli.json'
            subprocess.run([sys.executable,'detect.py','--data','data','--output',str(target)],check=True,capture_output=True)
            expected=json.loads(target.read_text())
        self.assertEqual(actual,expected)
        report=actual['development_verification']
        self.assertEqual([row['Episode'] for row in report],[f'TRN-{i:02}' for i in range(1,13)])
        self.assertTrue(all(row['Status']=='Matched' for row in report))
        for event in actual['events']:
            for member in event['members']:
                diagnostic=member['diagnostics']
                duration=(pd.Timestamp(member['end'])-pd.Timestamp(member['start'])).total_seconds()
                terms=score_terms(diagnostic['flag_fractions'],member['trigger'],duration,diagnostic['s4_median'],diagnostic['kp_median'],diagnostic['rfi_confidence'],actual['settings'])
                self.assertEqual({cause:round(max(0.,sum(t.values())),3) for cause,t in terms.items()},member['scores'])

if __name__=='__main__': unittest.main()
