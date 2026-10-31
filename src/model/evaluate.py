import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
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

REPORT_FILE = ARTIFACT_DIR / "evaluation_report.json"


# --------------------------------------------------
# 2. CALCULATE PERFORMANCE METRICS
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

    r2 = (
        r2_score(y_true, y_pred)
        if len(y_true) >= 2
        else None
    )

    # MAPE is unreliable when actual counts contain zeros.
    nonzero_mask = np.asarray(y_true) != 0

    if nonzero_mask.any():

        mape = mean_absolute_percentage_error(
            np.asarray(y_true)[nonzero_mask],
            np.asarray(y_pred)[nonzero_mask]
        ) * 100

    else:

        mape = None

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": (
            float(r2)
            if r2 is not None
            else None
        ),
        "MAPE_percent": (
            float(mape)
            if mape is not None
            else None
        )
    }


# --------------------------------------------------
# 3. LOAD TRAINED MODEL
# --------------------------------------------------

def load_trained_model():

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            "Model not found. Run train.py first."
        )

    bundle = joblib.load(
        MODEL_FILE
    )

    return bundle


# --------------------------------------------------
# 4. EVALUATE ON UNSEEN WARDS
# --------------------------------------------------

def evaluate_unseen_wards(
    model,
    X,
    y,
    groups
):

    unique_groups = pd.Series(
        groups
    ).nunique()

    if unique_groups < 2:

        raise ValueError(
            "At least two wards are required for evaluation."
        )

    splitter = GroupShuffleSplit(
        n_splits=min(5, unique_groups),
        test_size=0.2,
        random_state=42
    )

    fold_results = []

    all_actual = []
    all_predictions = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        current_model = clone(model)

        current_model.fit(
            X_train,
            y_train
        )

        predictions = current_model.predict(
            X_test
        )

        predictions = np.maximum(
            predictions,
            0
        )

        metrics = calculate_metrics(
            y_test,
            predictions
        )

        fold_results.append(metrics)

        all_actual.extend(
            y_test.tolist()
        )

        all_predictions.extend(
            predictions.tolist()
        )

        print(f"\nFold {fold}")

        for metric, value in metrics.items():

            print(f"{metric}: {value}")

    overall_metrics = calculate_metrics(
        all_actual,
        all_predictions
    )

    return {
        "fold_results": fold_results,
        "overall_metrics": overall_metrics,
        "actual": all_actual,
        "predicted": all_predictions
    }


# --------------------------------------------------
# 5. FEATURE IMPORTANCE
# --------------------------------------------------

def get_feature_importance(
    model,
    feature_names
):

    estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):

        importance = estimator.feature_importances_

    elif hasattr(estimator, "coef_"):

        importance = np.abs(
            estimator.coef_
        )

    else:

        return {}

    importance_df = pd.DataFrame({

        "feature": feature_names,

        "importance": importance

    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    return {
        row["feature"]: float(row["importance"])
        for _, row in importance_df.head(15).iterrows()
    }


# --------------------------------------------------
# 6. SAVE EVALUATION REPORT
# --------------------------------------------------

def save_report(report):

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REPORT_FILE,
        "w"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print(
        "\nEvaluation report saved:",
        REPORT_FILE
    )


# --------------------------------------------------
# 7. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("\n========== WARD PULSE MODEL EVALUATION ==========")

    # Load model.
    bundle = load_trained_model()

    model = bundle["model"]

    model_name = bundle["model_name"]

    feature_columns = bundle["feature_columns"]

    print("\nSelected model:", model_name)

    # Load datasets.
    training_data, prediction_data = (
        preprocess_datasets(
            WARD_FILE,
            COMPLAINT_FILE
        )
    )

    # Prepare features.
    X, y, X_predict, prediction_rows = (
        build_feature_matrices(
            training_data,
            prediction_data
        )
    )

    # Align features with the saved model.
    X = X.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Match ward IDs to valid target rows.
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

    # Evaluate on unseen wards.
    results = evaluate_unseen_wards(
        model,
        X,
        y,
        groups
    )

    # Feature importance from final trained model.
    importance = get_feature_importance(
        model,
        feature_columns
    )

    # Build report.
    report = {

        "model_name": model_name,

        "number_of_training_samples": len(X),

        "number_of_wards": int(
            groups.nunique()
        ),

        "number_of_features": len(feature_columns),

        "overall_metrics": results[
            "overall_metrics"
        ],

        "fold_results": results[
            "fold_results"
        ],

        "feature_importance": importance

    }

    save_report(report)

    # Display results.
    print("\n========== FINAL PERFORMANCE ==========")

    for metric, value in results[
        "overall_metrics"
    ].items():

        print(f"{metric}: {value}")

    print("\nTop Important Features:")

    for feature, value in list(
        importance.items()
    )[:10]:

        print(
            f"{feature}: {value:.4f}"
        )

    print("\n========== EVALUATION COMPLETED ==========")


# --------------------------------------------------
# 8. EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    main()