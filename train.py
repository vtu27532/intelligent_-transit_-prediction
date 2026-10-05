"""Compare two delay regressors and save the strongest one for the API."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from backend.model import CATEGORICAL_COLUMNS, DATA_PATH, FEATURE_COLUMNS, METRICS_PATH, MODEL_PATH, NUMERIC_COLUMNS
except ModuleNotFoundError:
    from model import CATEGORICAL_COLUMNS, DATA_PATH, FEATURE_COLUMNS, METRICS_PATH, MODEL_PATH, NUMERIC_COLUMNS


def make_pipeline(regressor) -> Pipeline:
    """Encode names as numeric columns and scale the small numeric inputs."""
    preparation = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ("numbers", StandardScaler(), NUMERIC_COLUMNS),
    ])
    return Pipeline([("prepare", preparation), ("regressor", regressor)])


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Trip data is missing. Run python generate_data.py first.")

    trips = pd.read_csv(DATA_PATH)
    features = trips[FEATURE_COLUMNS]
    target = trips["delay_minutes"]
    # Keep one fifth of trips unseen so the score reflects future predictions.
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    candidates = {
        "Random Forest": RandomForestRegressor(
            n_estimators=120, min_samples_leaf=2, max_features=0.9,
            random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=110, learning_rate=0.06, max_depth=3,
            loss="huber", random_state=42,
        ),
    }
    results = {}
    fitted = {}
    for name, regressor in candidates.items():
        # Each pipeline converts text fields into numbers before fitting the regressor.
        pipeline = make_pipeline(regressor)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        # Score both models against the same held-out trips for a fair comparison.
        results[name] = {
            "mae": round(float(mean_absolute_error(y_test, predictions)), 3),
            "r2": round(float(r2_score(y_test, predictions)), 4),
        }
        fitted[name] = pipeline
        print(f"{name}: MAE {results[name]['mae']:.2f} min | R2 {results[name]['r2']:.3f}")

    # Lower MAE means a smaller typical miss, so save that model for the API.
    best_name = min(results, key=lambda name: results[name]["mae"])
    best_model = fitted[best_name]
    # A 90th-percentile held-out error gives users a practical uncertainty band.
    residuals = np.abs(y_test.to_numpy() - best_model.predict(X_test))
    confidence_minutes = max(2.0, float(np.quantile(residuals, 0.9)))

    encoder = best_model.named_steps["prepare"].named_transformers_["categories"]
    feature_names = list(encoder.get_feature_names_out(CATEGORICAL_COLUMNS)) + NUMERIC_COLUMNS
    importance = best_model.named_steps["regressor"].feature_importances_
    grouped = {}
    for feature, score in zip(feature_names, importance):
        category = feature.split("_")[0] if feature not in NUMERIC_COLUMNS else feature
        grouped[category] = grouped.get(category, 0.0) + float(score)
    feature_importance = [
        {"feature": key.replace("traffic", "traffic").replace("day", "day of week").title(), "importance": round(value, 4)}
        for key, value in sorted(grouped.items(), key=lambda item: item[1], reverse=True)
    ]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    metadata = {
        "best_model": best_name,
        "best_mae": results[best_name]["mae"],
        "best_r2": results[best_name]["r2"],
        "confidence_minutes": round(confidence_minutes, 1),
        "model_comparison": results,
        "feature_importance": feature_importance,
        "training_rows": int(len(trips)),
    }
    METRICS_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {best_name} to {MODEL_PATH}")
    print(f"90% confidence band: +/- {confidence_minutes:.1f} minutes")


if __name__ == "__main__":
    main()
