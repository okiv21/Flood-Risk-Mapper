"""Score a complete, single issue-time batch with a trained research M3 artifact."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile

from floodrisk.scoring.windows import FEATURES, features, instant


def score(inputs, registry, model_dir, output):
    import lightgbm as lgb
    import numpy as np
    import pandas as pd

    report = json.loads((model_dir / "evaluation.json").read_text())
    model_path = model_dir / "event_model.txt"
    if (
        report["target"] != "reviewed flood onset in the next 72 complete hours"
        or report["features"] != FEATURES
        or hashlib.sha256(model_path.read_bytes()).hexdigest() != report["model_sha256"]
    ):
        raise ValueError("Unexpected M3 artifact")
    with inputs.open() as f:
        rows = list(csv.DictReader(f))
    with registry.open() as f:
        ids = {row["lga_id"] for row in csv.DictReader(f)}
    if (
        len(ids) != 774
        or len(rows) != len(ids)
        or {row["lga_id"] for row in rows} != ids
    ):
        raise ValueError("Require one row per registered LGA")
    if len({instant(row["issue_utc"]) for row in rows}) != 1:
        raise ValueError("Mix of issue times")
    frame = pd.DataFrame([features(row) for row in rows])[FEATURES]
    booster = lgb.Booster(model_file=str(model_path))
    if booster.feature_name() != FEATURES:
        raise ValueError("Model feature order mismatch")
    scores = np.interp(
        booster.predict(frame), report["isotonic"]["x"], report["isotonic"]["y"]
    )
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Invalid probabilities")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=output.parent, delete=False, newline=""
    ) as f:
        fields = [
            "lga_id",
            "issue_utc",
            "window_end_utc",
            "research_probability",
            "model_sha256",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row, s in zip(rows, scores):
            writer.writerow(
                dict(
                    lga_id=row["lga_id"],
                    issue_utc=row["issue_utc"],
                    window_end_utc=row["window_end_utc"],
                    research_probability=s,
                    model_sha256=report["model_sha256"],
                )
            )
    Path(f.name).replace(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["inputs", "registry", "model-dir", "output"]:
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    score(a.inputs, a.registry, a.model_dir, a.output)
