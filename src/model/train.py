import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from .preprocess import preprocess_datasets
from .feature_engineering import build_feature_matrices


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"

ARTIFACT_DIR = BASE_DIR / "artifacts"

WARD_FILE = DATA_DIR / "ward_features.csv"

COMPLAINT_FILE = DATA_DIR / "historical_complaints.csv"

MODEL_FILE = ARTIFACT_DIR / "best_model.joblib"

REPORT_FILE = ARTIFACT_DIR / "model_report.json"

RANDOM_STATE = 42

N_SPLITS = 5


# --------------------------------------------------
# 2. CREATE MODEL CANDIDATES
# --------------------------------------------------

def get_models():

    models = {

        "Ridge Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),

        "Extra Trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),

        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE
            ))
        ]),

        "Hist Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                max_iter=150,
                learning_rate=0.05,
                random_state=RANDOM_STATE
            ))
        ])
    }

    return models


# --------------------------------------------------
# 3. EVALUATION METRICS
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    # R2 is not meaningful for a single observation.
    r2 = (
        r2_score(y_true, y_pred)
        if len(y_true) >= 2
        else np.nan
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


# --------------------------------------------------
# 4. GROUP-BASED MODEL VALIDATION
# --------------------------------------------------

def evaluate_models(
    X,
    y,
    groups
):

    models = get_models()

    unique_groups = pd.Series(groups).nunique()

    if unique_groups < 2:

        raise ValueError(
            "At least two different wards are required "
            "for group-based validation."
        )

    # Split entire wards rather than individual rows.
    # This prevents the same ward appearing in both
    # training and validation sets.

    splitter = GroupShuffleSplit(
        n_splits=min(N_SPLITS, max(2, unique_groups)),
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    results = {}

    for model_name, model in models.items():

        print(f"\nEvaluating: {model_name}")

        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(X, y, groups),
            start=1
        ):

            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]

            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            current_model = clone(model)

            try:

                current_model.fit(
                    X_train,
                    y_train
                )

                predictions = current_model.predict(
                    X_val
                )

                metrics = calculate_metrics(
                    y_val,
                    predictions
                )

                fold_scores.append(metrics)

                print(
                    f"Fold {fold}: "
                    f"MAE={metrics['MAE']:.4f}, "
                    f"RMSE={metrics['RMSE']:.4f}"
                )

            except Exception as error:

                print(
                    f"Fold {fold} failed: {error}"
                )

        if not fold_scores:

            print(
                f"Skipping {model_name}: "
                "all validation folds failed."
            )

            continue

        results[model_name] = {

            "MAE": float(
                np.mean([
                    score["MAE"]
                    for score in fold_scores
                ])
            ),

            "RMSE": float(
                np.mean([
                    score["RMSE"]
                    for score in fold_scores
                ])
            ),

            "R2": float(
                np.nanmean([
                    score["R2"]
                    for score in fold_scores
                ])
            ),

            "successful_folds": len(fold_scores)

        }

    if not results:

        raise RuntimeError(
            "All candidate models failed."
        )

    return results


# --------------------------------------------------
# 5. SELECT BEST MODEL
# --------------------------------------------------

def select_best_model(results):

    best_model_name = min(
        results,
        key=lambda name: (
            results[name]["MAE"],
            results[name]["RMSE"]
        )
    )

    print("\n========== MODEL COMPARISON ==========")

    for name, scores in results.items():

        print(f"\n{name}")

        print(
            f"MAE: {scores['MAE']:.4f}"
        )

        print(
            f"RMSE: {scores['RMSE']:.4f}"
        )

        print(
            f"R2: {scores['R2']:.4f}"
        )

    print("\nBest Model:", best_model_name)

    print("======================================")

    return best_model_name


# --------------------------------------------------
# 6. TRAIN FINAL MODEL
# --------------------------------------------------

def train_final_model(
    model_name,
    X,
    y
):

    models = get_models()

    model = clone(
        models[model_name]
    )

    print(
        f"\nTraining final model: {model_name}"
    )

    model.fit(
        X,
        y
    )

    return model


# --------------------------------------------------
# 7. SAVE MODEL AND REPORT
# --------------------------------------------------

def save_model(
    model,
    model_name,
    results,
    feature_columns
):

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_bundle = {

        "model": model,

        "model_name": model_name,

        "feature_columns": list(feature_columns),

        "target": "complaint_count",

        "validation_results": results

    }

    joblib.dump(
        model_bundle,
        MODEL_FILE
    )

    report = {

        "best_model": model_name,

        "validation_results": results,

        "features": list(feature_columns),

        "target": "complaint_count"

    }

    with open(
        REPORT_FILE,
        "w"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print("\nModel saved:", MODEL_FILE)

    print("Report saved:", REPORT_FILE)


# --------------------------------------------------
# 8. MAIN TRAINING PIPELINE
# --------------------------------------------------

def main():

    print("\n========== WARD PULSE ML TRAINING ==========\n")

    # Check datasets.
    if not WARD_FILE.exists():

        raise FileNotFoundError(
            f"Ward dataset not found: {WARD_FILE}"
        )

    if not COMPLAINT_FILE.exists():

        raise FileNotFoundError(
            f"Complaint dataset not found: {COMPLAINT_FILE}"
        )

    # Step 1: Preprocessing.
    print("Step 1: Loading and preprocessing datasets...")

    training_data, prediction_data = (
        preprocess_datasets(
            WARD_FILE,
            COMPLAINT_FILE
        )
    )

    # Step 2: Feature engineering.
    print("\nStep 2: Creating model features...")

    X, y, X_predict, prediction_rows = (
        build_feature_matrices(
            training_data,
            prediction_data
        )
    )

    # Keep ward IDs aligned with the valid training rows.
    groups = (
        training_data.loc[
            y.index,
            "ward_id"
        ]
        .astype(str)
        .reset_index(drop=True)
    )

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    print("\nTraining samples:", len(X))

    print("Prediction samples:", len(X_predict))

    print("Number of features:", X.shape[1])

    print("Unique training wards:", groups.nunique())

    # Step 3: Evaluate candidate models.
    print("\nStep 3: Comparing regression algorithms...")

    results = evaluate_models(
        X,
        y,
        groups
    )

    # Step 4: Select best model.
    print("\nStep 4: Selecting best model...")

    best_model_name = select_best_model(
        results
    )

    # Step 5: Train selected model on all training data.
    print("\nStep 5: Training final model...")

    final_model = train_final_model(
        best_model_name,
        X,
        y
    )

    # Step 6: Save model.
    print("\nStep 6: Saving trained model...")

    save_model(
        final_model,
        best_model_name,
        results,
        X.columns
    )

    print("\n========== TRAINING COMPLETED ==========")

    print("Selected model:", best_model_name)

    print("Model location:", MODEL_FILE)

    print("========================================")


# --------------------------------------------------
# 9. EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    main()