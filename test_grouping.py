"""Grouping invariants independent of labels or expected scenario counts."""
import unittest
import numpy as np
import pandas as pd
import detect

class Grouping(unittest.TestCase):
    def candidate(self, a,b,tractor='T1',cause='JAMMING'):
        start=pd.Timestamp('2026-05-04T12:00Z')
        return {'start':start+pd.Timedelta(seconds=a),'end':start+pd.Timedelta(seconds=b),'tractors':[tractor],'class':cause,'trigger':'high_error'}

    def test_components_and_original_boundaries(self):
        # Interleaved tractor T2 must not prevent T1's fragments joining.
        items=[self.candidate(0,10),self.candidate(15,20,'T2'),self.candidate(25,35)]
        result=detect.merge_candidates(items,20)
        self.assertEqual(len(result),2)
        self.assertEqual(result[0]['members'],[items[0],items[2]])
        self.assertEqual(items[0]['end'],pd.Timestamp('2026-05-04T12:00:10Z'))
        result=detect.merge_candidates(items+[self.candidate(0,35,cause='SPOOFING')],20)
        self.assertEqual(len(result),3)

    def test_missing_and_parked_samples_prevent_gap_merging(self):
        items=[self.candidate(0,10),self.candidate(15,20)]
        all_samples=pd.date_range('2026-05-04T12:00Z',periods=20,freq='s').astype('int64').to_numpy()
        self.assertEqual(len(detect.merge_candidates(items,10,{'T1':all_samples})),1)
        self.assertEqual(len(detect.merge_candidates(items,10,{'T1':np.delete(all_samples,12)})),2)
        self.assertEqual(len(detect.merge_candidates(items,10,{})),2)
        self.assertEqual(len(detect.merge_candidates([items[0],self.candidate(15,20,'T2')],10)),2)

    def test_transitive_bridge_and_gap_limit(self):
        items=[self.candidate(0,10),self.candidate(40,50),self.candidate(15,45,'T2')]
        # The T2 candidate overlaps the later T1 candidate, and is near but
        # not overlapping the first one. It must not bridge unrelated tractors.
        self.assertEqual(len(detect.merge_candidates(items,5)),2)
        items.append(self.candidate(10,16))
        self.assertEqual(len(detect.merge_candidates(items,5)),1)

    def test_duplicate_policy_preserves_independent_errors(self):
        multi=self.candidate(0,17,cause='MULTIPATH')
        spoof=self.candidate(0,17,cause='SPOOFING')
        spoof['diagnostics']={'flag_fractions':{'signal_spread':1.,'excess_error':0.}}
        kept,suppressed=detect.resolve_short_duplicates([multi,spoof],detect.DEFAULTS)
        self.assertEqual(kept,[multi]);self.assertEqual(len(suppressed),1)
        spoof['diagnostics']['flag_fractions']['excess_error']=1.
        kept,_=detect.resolve_short_duplicates([multi,spoof],detect.DEFAULTS)
        self.assertEqual(len(kept),2)
        spoof['tractors']=['T2']
        kept,_=detect.resolve_short_duplicates([multi,spoof],detect.DEFAULTS)
        self.assertEqual(len(kept),2)

if __name__=='__main__': unittest.main()
