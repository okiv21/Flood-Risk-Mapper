"""Create an unlabelled chip review queue and gate reviewed scene-condition labels."""
from datetime import datetime


def review_queue(pair,chip_size=256):
    if chip_size<1:raise ValueError('Positive chip size required')
    event=pair['acquisitions']['event_candidate'];reference=pair['acquisitions']['reference_candidate']
    grid=event['download_grid']
    if grid!=reference['download_grid']:raise ValueError('Pair grids differ')
    for k in ['relativeOrbitNumber_start','orbitProperties_pass']:
        if event['source'][k]!=reference['source'][k]:raise ValueError('Pair orbits differ')
    t=grid['crs_transform'];w,h=grid['dimensions']
    if t[1] or t[3]:raise ValueError('North-up review grid required')
    rows=[]
    for row in range(0,h,chip_size):
        for col in range(0,w,chip_size):
            width,height=min(chip_size,w-col),min(chip_size,h-row)
            rows.append(dict(chip_id=f"{pair['case']['case_id']}_{row}_{col}",group_id=pair['case']['group_id'],
                col_off=col,row_off=row,width=width,height=height,crs=grid['crs'],
                left=t[2]+col*t[0],top=t[5]+row*t[4],
                right=t[2]+(col+width)*t[0],bottom=t[5]+(row+height)*t[4],
                event_scene=event['source']['system:index'],reference_scene=reference['source']['system:index'],
                status='pending',label='',reviewer='',evidence_type='',evidence_uri='',
                observed_start_utc='',observed_end_utc='',reviewed_at_utc='',
                target_scope='scene_condition_only',notes='No automatic flood/non-flood label; reference is not assumed dry'))
    return rows


def eligible_scene_labels(rows):
    """Accept only explicit reviewed scene labels; this cannot verify external evidence itself."""
    accepted=[]
    for row in rows:
        if row['status']!='reviewed':continue
        if row['label'] not in ['flood','non_flood']:continue
        if row.get('evidence_type') not in ['manual_annotation','field_observation','independent_reference']:
            raise ValueError('An independent evidence type is required; model output alone is insufficient')
        for key in ['group_id','reviewer','evidence_uri','observed_start_utc','observed_end_utc','reviewed_at_utc']:
            if not row.get(key):raise ValueError(f'Missing review field: {key}')
        times=[datetime.fromisoformat(row[k].replace('Z','+00:00')) for k in ['observed_start_utc','observed_end_utc','reviewed_at_utc']]
        if any(t.tzinfo is None for t in times) or times[0]>times[1] or times[2]<times[1]:
            raise ValueError('Invalid review/evidence timestamps')
        if row.get('target_scope')!='scene_condition_only':raise ValueError('72-hour event labels require a separate protocol')
        accepted.append(row)
    return accepted
