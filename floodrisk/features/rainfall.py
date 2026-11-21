"""Validate and aggregate rainfall without crossing the declared time windows."""
from datetime import date, datetime, timedelta, timezone
import math


def nonnegative(value):
    if value is None or isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Rainfall must be finite nonnegative numbers; missing values are not zero')
    return float(value)


def antecedent_features(records, end_date):
    """14 complete UTC dates preceding end_date, excluding the acquisition day."""
    end=date.fromisoformat(end_date)
    expected=[end-timedelta(days=i) for i in range(14,0,-1)]
    dates=[date.fromisoformat(r['date']) for r in records]
    if dates!=expected:raise ValueError('Require exactly 14 ordered consecutive dates before end_date')
    values=[nonnegative(r['mm']) for r in records]
    return {**{f'past_{n}d_mm':sum(values[-n:]) for n in [1,3,7,14]},'max_daily_14d_mm':max(values)}


def forecast_features(payload, retrieved_at):
    """72 complete future one-hour periods, with precipitation timestamped at period end."""
    if retrieved_at.tzinfo is None:raise ValueError('Timezone-aware retrieval time required')
    if payload.get('utc_offset_seconds')!=0 or payload['hourly_units'].get('precipitation')!='mm':
        raise ValueError('Expected UTC precipitation in mm')
    times=payload['hourly']['time'];values=payload['hourly']['precipitation']
    if len(times)!=len(values):raise ValueError('Time/value lengths differ')
    parsed=[datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in times]
    if len(set(parsed))!=len(parsed):raise ValueError('Duplicate hourly timestamps')
    now=retrieved_at.astimezone(timezone.utc)
    start=now.replace(minute=0,second=0,microsecond=0)
    if start<now:start+=timedelta(hours=1)
    by_time=dict(zip(parsed,values))
    periods=[]
    for hour in range(1,73):
        end=start+timedelta(hours=hour)
        if end not in by_time:raise ValueError('Incomplete 72-hour forecast')
        periods.append(dict(period_end_utc=end.isoformat(),mm=nonnegative(by_time[end])))
    features={f'next_{n}h_mm':sum(r['mm'] for r in periods[:n]) for n in [24,48,72]}
    features['max_hourly_72h_mm']=max(r['mm'] for r in periods)
    return dict(window_start_utc=start.isoformat(),window_end_utc=(start+timedelta(hours=72)).isoformat(),
                retrieved_at_utc=now.isoformat(),features=features,periods=periods,
                convention='72 complete hours beginning at the next UTC hour; source timestamps mark preceding-hour totals')
