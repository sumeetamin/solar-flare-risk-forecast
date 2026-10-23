"""Research baseline for next-24-hour solar flare count estimation.

Dataset: UCI Solar Flare (id 89). This dataset is untimestamped; the random
holdout below is not a temporal validation and must not be used operationally.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from ucimlrepo import fetch_ucirepo


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = fetch_ucirepo(id=89)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    y = y.apply(pd.to_numeric, errors="coerce")
    valid = y.notna().all(axis=1)
    X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)
    if X.empty or y.shape[1] != 3:
        raise ValueError(f"Expected three flare count targets; received {y.shape}.")
    return X, y


def train(output_dir: Path) -> dict[str, object]:
    X, y = load_data()
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    preprocessing = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    model = Pipeline(
        [
            ("preprocess", preprocessing),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=500,
                    min_samples_leaf=2,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model.fit(X_train, y_train)
    prediction = np.clip(model.predict(X_test), 0, None)
    metrics = {
        "dataset": "UCI Solar Flare (id 89)",
        "rows": int(len(X)),
        "features": list(X.columns),
        "targets": list(y.columns),
        "split": "random 75/25 holdout; seed=42; no timestamps available",
        "model": "RandomForestRegressor(n_estimators=500, min_samples_leaf=2)",
        "per_target": {
            target: {
                "mae": float(mean_absolute_error(y_test[target], prediction[:, i])),
                "median_absolute_error": float(
                    median_absolute_error(y_test[target], prediction[:, i])
                ),
                "r2": float(r2_score(y_test[target], prediction[:, i])),
            }
            for i, target in enumerate(y.columns)
        },
        "warning": "Research benchmark only; random holdout is not temporal validation or operational forecast skill.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.DataFrame(prediction, columns=[f"predicted_{c}" for c in y.columns])
    for target in y.columns:
        pred[f"actual_{target}"] = y_test[target].to_numpy()
    pred.to_csv(output_dir / "holdout_predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    importances = model.named_steps["regressor"].feature_importances_
    pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False
    ).to_csv(output_dir / "feature_importances.csv", index=False)
    joblib.dump(model, output_dir / "model.joblib")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    metrics = train(args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

