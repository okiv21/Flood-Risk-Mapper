"""Check time boundaries and missing-data rejection for rainfall features."""
from datetime import date,datetime,timedelta,timezone
import unittest
from floodrisk.features.rainfall import antecedent_features,forecast_features


class RainfallFeatureTests(unittest.TestCase):
    def test_antecedent_windows_exclude_acquisition_day(self):
        end=date(2022,8,12)
        rows=[dict(date=(end-timedelta(days=14-i)).isoformat(),mm=i+1) for i in range(14)]
        f=antecedent_features(rows,end.isoformat())
        self.assertEqual(f['past_1d_mm'],14)
        self.assertEqual(f['past_3d_mm'],39)
        self.assertEqual(f['past_14d_mm'],105)
        rows[-1]['date']=end.isoformat()
        with self.assertRaises(ValueError):antecedent_features(rows,end.isoformat())

    def payload(self):
        start=datetime(2026,11,20,tzinfo=timezone.utc)
        return dict(utc_offset_seconds=0,hourly_units={'precipitation':'mm'},hourly=dict(
            time=[(start+timedelta(hours=i)).strftime('%Y-%m-%dT%H:%M') for i in range(100)],
            precipitation=[1]*100))

    def test_forecast_uses_only_complete_future_hours(self):
        payload=self.payload()
        payload['hourly']['precipitation'][:12]=[999]*12
        result=forecast_features(payload,datetime(2026,11,20,10,30,tzinfo=timezone.utc))
        self.assertEqual(result['window_start_utc'],'2026-11-20T11:00:00+00:00')
        self.assertEqual(result['periods'][0]['period_end_utc'],'2026-11-20T12:00:00+00:00')
        self.assertEqual(result['features']['next_24h_mm'],24)
        self.assertEqual(result['features']['next_72h_mm'],72)
        self.assertEqual(len(result['periods']),72)

    def test_missing_or_duplicate_forecast_cannot_become_zero(self):
        now=datetime(2026,11,20,10,tzinfo=timezone.utc)
        for value in [None,-1,float('nan')]:
            payload=self.payload();payload['hourly']['precipitation'][11]=value
            with self.assertRaises(ValueError):forecast_features(payload,now)
        payload=self.payload();payload['hourly']['time'][11]=payload['hourly']['time'][12]
        with self.assertRaises(ValueError):forecast_features(payload,now)
        payload=self.payload();payload['hourly']['time']=payload['hourly']['time'][:20];payload['hourly']['precipitation']=payload['hourly']['precipitation'][:20]
        with self.assertRaises(ValueError):forecast_features(payload,now)

    def test_missing_historical_rain_is_rejected(self):
        rows=[dict(date=(date(2022,7,29)+timedelta(days=i)).isoformat(),mm=1) for i in range(14)]
        rows[0]['mm']=None
        with self.assertRaises(ValueError):antecedent_features(rows,'2022-08-12')
