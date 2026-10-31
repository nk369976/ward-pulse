import joblib
import pandas as pd
import numpy as np

from pathlib import Path

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

OUTPUT_FILE = DATA_DIR / "ward_predictions.csv"


# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------

def load_model():

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            "Trained model not found. Run train.py first."
        )

    bundle = joblib.load(MODEL_FILE)

    print(
        "Loaded model:",
        bundle["model_name"]
    )

    return bundle


# --------------------------------------------------
# 3. GENERATE PREDICTIONS
# --------------------------------------------------

def generate_predictions(
    model,
    X_predict
):

    predictions = model.predict(
        X_predict
    )

    # Complaint counts cannot be negative.
    predictions = np.maximum(
        predictions,
        0
    )

    return predictions


# --------------------------------------------------
# 4. ADD PREDICTIONS TO WARD INFORMATION
# --------------------------------------------------

def prepare_output(
    prediction_rows,
    predictions
):

    output = prediction_rows.copy()

    output["predicted_complaints"] = predictions

    # Expected counts can be fractional.
    # Round only when producing a whole-number estimate.
    output["predicted_complaints"] = (
        output["predicted_complaints"].round(2)
    )

    return output


# --------------------------------------------------
# 5. CREATE WARD-WISE SUMMARY
# --------------------------------------------------

def create_ward_summary(predictions):

    summary = (
        predictions
        .groupby("ward_id", as_index=False)
        ["predicted_complaints"]
        .sum()
    )

    summary = summary.rename(
        columns={
            "predicted_complaints":
            "total_predicted_complaints"
        }
    )

    summary = summary.sort_values(
        by="total_predicted_complaints",
        ascending=False
    )

    return summary


# --------------------------------------------------
# 6. SAVE PREDICTIONS
# --------------------------------------------------

def save_predictions(
    predictions,
    summary
):

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions.to_csv(
        OUTPUT_FILE,
        index=False
    )

    summary.to_csv(
        DATA_DIR / "ward_summary.csv",
        index=False
    )

    print("\nPredictions saved:", OUTPUT_FILE)

    print(
        "Ward summary saved:",
        DATA_DIR / "ward_summary.csv"
    )


# --------------------------------------------------
# 7. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("\n========== WARD PULSE PREDICTION ==========\n")

    # Load trained model.
    bundle = load_model()

    model = bundle["model"]

    feature_columns = bundle["feature_columns"]

    # Load and preprocess datasets.
    training_data, prediction_data = (
        preprocess_datasets(
            WARD_FILE,
            COMPLAINT_FILE
        )
    )

    # Generate feature matrices.
    X_train, y_train, X_predict, prediction_rows = (
        build_feature_matrices(
            training_data,
            prediction_data
        )
    )

    # Ensure prediction columns match training columns.
    X_predict = X_predict.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Predict complaint demand.
    predicted_counts = generate_predictions(
        model,
        X_predict
    )

    # Prepare detailed output.
    predictions = prepare_output(
        prediction_rows,
        predicted_counts
    )

    # Create ward-wise summary.
    summary = create_ward_summary(
        predictions
    )

    # Save results.
    save_predictions(
        predictions,
        summary
    )

    print("\nTop 10 wards by predicted demand:")

    print(
        summary.head(10).to_string(index=False)
    )

    print("\n========== PREDICTION COMPLETED ==========")


# --------------------------------------------------
# 8. EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    main()