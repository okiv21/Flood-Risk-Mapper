import unittest
from floodrisk.features.evidence import review_queue,eligible_scene_labels


class EvidenceReviewTests(unittest.TestCase):
    def pair(self):
        scene=dict(download_grid={'dimensions':[513,257],'crs':'EPSG:32632',
            'crs_transform':[10,0,1000,0,-10,5000]},source={'relativeOrbitNumber_start':1,'orbitProperties_pass':'ASCENDING','system:index':'example'})
        return dict(case={'case_id':'example','group_id':'one_event'},acquisitions={'event_candidate':scene,'reference_candidate':scene})

    def test_queue_covers_edges_and_never_generates_labels(self):
        rows=review_queue(self.pair())
        self.assertEqual(len(rows),6)
        self.assertEqual(sum(r['width']*r['height'] for r in rows),513*257)
        self.assertEqual(rows[-1]['width'],1)
        self.assertEqual(rows[-1]['height'],1)
        self.assertEqual({r['group_id'] for r in rows},{'one_event'})
        self.assertEqual(eligible_scene_labels(rows),[])

    def test_review_requires_independent_evidence_and_times(self):
        row=review_queue(self.pair())[0]
        row.update(status='reviewed',label='flood',evidence_type='model_output')
        with self.assertRaises(ValueError):eligible_scene_labels([row])
        row.update(evidence_type='manual_annotation',reviewer='test_reviewer',evidence_uri='test://annotation',
            observed_start_utc='2022-10-13T17:00:00Z',observed_end_utc='2022-10-13T18:00:00Z',reviewed_at_utc='2022-10-14T00:00:00Z')
        self.assertEqual(len(eligible_scene_labels([row])),1)
        row['observed_end_utc']='2022-10-13T18:00:00'
        with self.assertRaises(ValueError):eligible_scene_labels([row])
