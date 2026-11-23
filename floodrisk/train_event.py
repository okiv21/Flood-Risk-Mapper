"""Fit a research-only 72-hour classifier when reviewed, as-issued records exist."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from floodrisk.scoring.windows import FEATURES, features, instant, outcome


def cohort(row):
    start, end = instant(row["issue_utc"]), instant(row["window_end_utc"])
    for name, first, last in [
        ("train", 2016, 2022),
        ("validation", 2022, 2023),
        ("test", 2023, 9999),
    ]:
        if datetime(first, 1, 1, tzinfo=timezone.utc) <= start and end <= datetime(
            last, 1, 1, tzinfo=timezone.utc
        ):
            return name
    return None


def run(source, output):
    rows = {name: [] for name in ["train", "validation", "test"]}
    excluded = 0
    seen = set()
    evidence_sets = {}
    with source.open() as f:
        for row in csv.DictReader(f):
            label = outcome(row)
            split = cohort(row)
            if label is None or split is None:
                excluded += 1
                continue
            x = features(row)
            key = (row["lga_id"], instant(row["issue_utc"]))
            if key in seen:
                raise ValueError("Duplicate LGA/issue record")
            seen.add(key)
            if evidence_sets.setdefault(row["evidence_id"], split) != split:
                raise ValueError("Evidence shared across temporal splits")
            rows[split].append({**x, "target": label})
    if any({r["target"] for r in records} != {0, 1} for records in rows.values()):
        raise ValueError("Each time split needs reviewed onset and event-free examples")
    train, val, test = [pd.DataFrame(rows[k]) for k in rows]
    model = LGBMClassifier(
        n_estimators=100, num_leaves=15, random_state=42, n_jobs=2, verbosity=-1
    )
    model.fit(train[FEATURES], train.target)
    calibration = IsotonicRegression(out_of_bounds="clip").fit(
        model.predict_proba(val[FEATURES])[:, 1], val.target
    )
    pred = calibration.predict(model.predict_proba(test[FEATURES])[:, 1])
    output.mkdir(parents=True, exist_ok=True)
    path = output / "event_model.txt"
    model.booster_.save_model(str(path))
    report = dict(
        target="reviewed flood onset in the next 72 complete hours",
        use="research_only",
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        features=FEATURES,
        isotonic=dict(
            x=calibration.X_thresholds_.tolist(), y=calibration.y_thresholds_.tolist()
        ),
        rows={k: len(v) for k, v in rows.items()},
        excluded=excluded,
        test=dict(
            roc_auc=roc_auc_score(test.target, pred),
            average_precision=average_precision_score(test.target, pred),
            brier=brier_score_loss(test.target, pred),
            constant_training_rate_brier=brier_score_loss(
                test.target, np.full(len(test), train.target.mean())
            ),
        ),
    )
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["source", "output"]:
        p.add_argument("--" + key, required=True, type=Path)
    a = p.parse_args()
    run(a.source, a.output)
