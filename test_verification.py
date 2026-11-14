import unittest
import pandas as pd
import detect
from verification import development_report

class Verification(unittest.TestCase):
    def label(self,identifier='TRN-01',start='2026-05-04T12:00Z',end='2026-05-04T12:10Z'):
        return {'id':identifier,'start':start,'end':end,'class':'JAMMING','tractors':['T1']}
    def event(self, label):
        member={**label,'start':pd.Timestamp(label['start']),'end':pd.Timestamp(label['end'])}
        return {**member,'members':[member]}
    def test_one_to_one_and_no_wrong_class_or_tractor(self):
        label=self.label();event=self.event(label)
        rows=development_report([event],[label,{**label,'id':'TRN-02'}])
        self.assertEqual(sum(r['Status']=='Matched' for r in rows),1)
        wrong={**event,'class':'SPOOFING'}
        self.assertEqual(development_report([wrong],[label])[0]['Status'],'Missed')
        wrong={**event,'tractors':['T2'],'members':[{**event['members'][0],'tractors':['T2']}]}
        self.assertEqual(development_report([wrong],[label])[0]['Status'],'Missed')
    def test_tiny_overlap_and_evaluation_are_not_matches(self):
        label=self.label();event=self.event(label)
        event['end']=pd.Timestamp(label['start'])+pd.Timedelta(seconds=1)
        self.assertEqual(development_report([event],[label])[0]['Status'],'Missed')
        evaluation=self.label('EVAL','2026-05-06T12:00Z','2026-05-06T13:00Z')
        self.assertEqual(development_report([self.event(evaluation)],[evaluation]),[])
    def test_brief_multipath_stays_per_tractor(self):
        label=self.label(start='2026-05-04T12:00Z',end='2026-05-04T12:00:17Z')
        label['class']='MULTIPATH'
        member=self.event(label)['members'][0]
        other={**member,'tractors':['T2']}
        later={**member,'start':member['start']+pd.Timedelta(seconds=30),'end':member['end']+pd.Timedelta(seconds=30)}
        events=detect.merge_candidates([member,other,later],180)
        self.assertEqual(len(events),3)
        report=development_report(events,[label])
        self.assertEqual(report[0]['Status'],'Matched')
        self.assertEqual(report[0]['Time IoU'],1.)
        self.assertEqual(report[0]['Tractor IoU'],1.)

if __name__=='__main__':unittest.main()
