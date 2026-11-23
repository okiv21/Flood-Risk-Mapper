"""Fit and evaluate a retrospective terrain benchmark on Nigerian GFD event samples."""

import argparse
import hashlib
import json
from pathlib import Path

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

FEATURES = [
    "elevation_m",
    "slope_degrees",
    "adjusted_elevation_m",
    "log1p_upstream_area_km2",
]


def fit(frame):
    model = LGBMClassifier(
        n_estimators=100,
        num_leaves=15,
        learning_rate=0.05,
        random_state=42,
        n_jobs=2,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    model.fit(
        frame[FEATURES],
        frame.target,
        sample_weight=frame.sample_weight / frame.sample_weight.mean(),
    )
    return model


def metrics(frame, scores):
    return {
        name: float(fn(frame.target, scores, sample_weight=frame.sample_weight))
        for name, fn in [
            ("roc_auc", roc_auc_score),
            ("average_precision", average_precision_score),
            ("brier", brier_score_loss),
        ]
    }


def prevalence(frame):
    return float(np.average(frame.target, weights=frame.sample_weight))


def run(directory, output):
    inventory = json.loads((directory / "inventory.json").read_text())
    frames = []
    for event in inventory["event_inventory"]:
        if "samples" not in event:
            continue
        path = directory / f"event_{event['metadata']['id']}.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != event["sha256"]:
            raise ValueError(f"Changed sample file: {path}")
        frames.append(pd.read_csv(path))
    data = pd.concat(frames, ignore_index=True)
    if (
        len(data) != inventory["samples"]
        or not np.isfinite(data[FEATURES + ["sample_weight"]]).all().all()
        or not (data.sample_weight > 0).all()
        or not np.allclose(data.groupby("event_id").sample_weight.sum(), 1)
    ):
        raise ValueError("Incomplete event sample or invalid design weights")
    years = pd.to_datetime(data.event_start_utc, utc=True).dt.year
    train = data[years <= 2009].reset_index(drop=True)
    valid = data[years.isin([2012, 2015])]
    test = data[years == 2018]
    if len(train) + len(valid) + len(test) != len(data) or any(
        set(f.target) != {0, 1} for f in [train, valid, test]
    ):
        raise ValueError(
            "Expected the predefined development, calibration and test cohorts with both outcomes"
        )
    folds = []
    oof = np.full(len(train), np.nan)
    for fit_ids, check_ids in GroupKFold(5).split(train, groups=train.block_50km):
        subset = train.iloc[check_ids]
        model = fit(train.iloc[fit_ids])
        scores = model.predict_proba(subset[FEATURES])[:, 1]
        oof[check_ids] = scores
        folds.append(
            dict(
                blocks=sorted(subset.block_50km.unique().tolist()),
                rows=len(subset),
                **metrics(subset, scores),
            )
        )
    model = fit(train)
    valid_scores = model.predict_proba(valid[FEATURES])[:, 1]
    calibration = IsotonicRegression(out_of_bounds="clip").fit(
        valid_scores, valid.target, sample_weight=valid.sample_weight
    )
    test_scores = model.predict_proba(test[FEATURES])[:, 1]
    scores = calibration.predict(test_scores)
    comparison = {
        "training_prevalence": metrics(test, np.full(len(test), prevalence(train))),
        "terrain_gbm": metrics(test, test_scores),
        "calibrated_terrain_gbm": metrics(test, scores),
    }
    scaler = StandardScaler().fit(train[FEATURES], sample_weight=train.sample_weight)
    logistic = LogisticRegression().fit(
        scaler.transform(train[FEATURES]),
        train.target,
        sample_weight=train.sample_weight / train.sample_weight.mean(),
    )
    comparison["logistic_terrain"] = metrics(
        test, logistic.predict_proba(scaler.transform(test[FEATURES]))[:, 1]
    )
    # Lower elevation alone gives a simple, training-free ranking for comparison.
    low_elevation = -test.elevation_m.to_numpy()
    ranking = {
        name: fn(test.target, low_elevation, sample_weight=test.sample_weight)
        for name, fn in [
            ("roc_auc", roc_auc_score),
            ("average_precision", average_precision_score),
        ]
    }
    output.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(output / "terrain_model.txt"))
    report = dict(
        target=inventory["settings"]["target"],
        features=FEATURES,
        model_sha256=hashlib.sha256(
            (output / "terrain_model.txt").read_bytes()
        ).hexdigest(),
        weighting=inventory["settings"]["weighting"],
        partitions={
            name: dict(
                events=sorted(f.event_id.unique().tolist()),
                rows=len(f),
                weighted_reference_fraction=prevalence(f),
            )
            for name, f in [("train", train), ("calibration", valid), ("test", test)]
        },
        spatial_validation=dict(
            method="Five disjoint 50 km block folds within the training years",
            folds=folds,
            pooled=metrics(train, oof),
        ),
        test=comparison,
        test_low_elevation_ranking=ranking,
        isotonic=dict(
            x=calibration.X_thresholds_.tolist(), y=calibration.y_thresholds_.tolist()
        ),
        use="Research only. Estimates event-conditional mapped water from retrospective terrain features. No issue-time, annual, or LGA warning probability.",
    )
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "source_inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.samples, args.output)["test"], indent=2))
