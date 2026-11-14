"""Submission contract and evaluation boundary regression tests."""
import unittest
import pandas as pd
from detect import build_submission
from submission.validate_submission import validate


class SubmissionTests(unittest.TestCase):
    def event(self):
        def member(start, end, tractor, evidence):
            return dict(start=pd.Timestamp(start), end=pd.Timestamp(end),
                        tractors=[tractor], evidence=[evidence])
        return dict(
            start=pd.Timestamp('2026-05-05T23:59Z'),
            end=pd.Timestamp('2026-05-07T00:01Z'), class_='JAMMING',
            members=[member('2026-05-05T23:59Z', '2026-05-06T00:00Z', 'T3', 'Development only'),
                     member('2026-05-06T00:00Z', '2026-05-06T00:01Z', 'T1', 'Signal loss'),
                     member('2026-05-06T23:59Z', '2026-05-07T00:01Z', 'T2', 'Fleet evidence')])

    def build(self, event, handle='BodenEngebretson'):
        event['class'] = event.pop('class_', 'JAMMING')
        return build_submission([event], handle,
                                'https://github.com/BodenEngebretson/Straight-Rows')

    def test_clips_boundaries_and_preserves_relevant_evidence(self):
        result = self.build(self.event())
        self.assertEqual(validate(result), [])
        detection = result['detections'][0]
        self.assertEqual(detection['start'], '2026-05-06T00:00:00+00:00')
        self.assertEqual(detection['end'], '2026-05-07T00:00:00+00:00')
        self.assertEqual(detection['tractors'], ['T1', 'T2'])
        self.assertNotIn('Development only', detection['evidence'])
        self.assertIn('Fleet evidence', detection['evidence'])
        self.assertEqual(detection['action'], 'HALT_AND_REPORT')

    def test_limit_preserves_operational_events(self):
        event = self.event()
        event['class'] = 'MULTIPATH'
        operational = dict(event, **{'class': 'JAMMING'})
        result = build_submission([event] * 514 + [operational] * 7,
                                  'BodenEngebretson',
                                  'https://github.com/BodenEngebretson/Straight-Rows')
        self.assertEqual(len(result['detections']), 500)
        self.assertEqual(sum(d['class'] == 'JAMMING' for d in result['detections']), 7)
        self.assertEqual(validate(result), [])

    def test_rejects_invalid_participant(self):
        with self.assertRaisesRegex(ValueError, 'github_handle'):
            self.build(self.event(), 'your-github-username')

    def test_rejects_empty_evaluation(self):
        with self.assertRaisesRegex(ValueError, 'non-empty'):
            build_submission([], 'BodenEngebretson',
                             'https://github.com/BodenEngebretson/Straight-Rows')

    def test_rejects_event_without_overlapping_members(self):
        event = self.event()
        event['members'] = event['members'][:1]
        with self.assertRaisesRegex(ValueError, 'overlapping member'):
            self.build(event)
