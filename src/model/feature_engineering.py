import numpy as np
import pandas as pd
import re


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

TARGET_COLUMN = "complaint_count"

ID_COLUMNS = [
    "ward_id",
    "ward_name",
    "city",
    "zone",
    "month",
    "category",
    "date",
    "complaint_count"
]

# Features that should not be used directly as model inputs.
# Add additional identifiers here if your dataset contains them.
EXCLUDED_COLUMNS = set(ID_COLUMNS)


# --------------------------------------------------
# 2. MONTH FEATURE EXTRACTION
# --------------------------------------------------

def extract_month(value):

    if pd.isna(value):
        return 0

    value = str(value).strip().lower()

    # Handle YYYY-MM and YYYY-MM-DD.
    match = re.match(r"^\d{4}-(\d{1,2})", value)

    if match:
        month = int(match.group(1))
        return month if 1 <= month <= 12 else 0

    # Handle month_01, month_02, etc.
    match = re.match(r"^month_(\d{1,2})$", value)

    if match:
        month = int(match.group(1))
        return month if 1 <= month <= 12 else 0

    # Handle numeric month values.
    if value.isdigit():

        month = int(value)

        if 1 <= month <= 12:
            return month

    # Handle month names and other date formats.
    date = pd.to_datetime(
        value,
        errors="coerce"
    )

    if pd.notna(date):
        return date.month

    return 0


# --------------------------------------------------
# 3. CALENDAR FEATURES
# --------------------------------------------------

def add_calendar_features(df):

    df = df.copy()

    if "month" in df.columns:

        df["month_number"] = df["month"].apply(
            extract_month
        )

        # Cyclical encoding:
        # December and January should be close in value.
        df["month_sin"] = np.sin(
            2 * np.pi * df["month_number"] / 12
        )

        df["month_cos"] = np.cos(
            2 * np.pi * df["month_number"] / 12
        )

    else:

        df["month_number"] = 0
        df["month_sin"] = 0
        df["month_cos"] = 0

    return df


# --------------------------------------------------
# 4. INFRASTRUCTURE FEATURES
# --------------------------------------------------

def add_infrastructure_features(df):

    df = df.copy()

    # Population density.
    if (
        "population" in df.columns
        and "area_sq_km" in df.columns
    ):

        area = df["area_sq_km"].replace(
            0, np.nan
        )

        calculated_density = (
            df["population"] / area
        )

        if "population_density" in df.columns:

            df["population_density"] = (
                df["population_density"]
                .fillna(calculated_density)
            )

        else:

            df["population_density"] = (
                calculated_density
            )

    # Road length per square kilometre.
    if (
        "road_length_km" in df.columns
        and "area_sq_km" in df.columns
    ):

        area = df["area_sq_km"].replace(
            0, np.nan
        )

        df["road_density"] = (
            df["road_length_km"] / area
        )

    # Streetlights per 1000 people.
    if (
        "streetlights" in df.columns
        and "population" in df.columns
    ):

        population = df["population"].replace(
            0, np.nan
        )

        df["streetlights_per_1000_people"] = (
            df["streetlights"] * 1000 / population
        )

    # Parks per 1000 people.
    if (
        "parks" in df.columns
        and "population" in df.columns
    ):

        population = df["population"].replace(
            0, np.nan
        )

        df["parks_per_1000_people"] = (
            df["parks"] * 1000 / population
        )

    # Lakes per square kilometre.
    if (
        "lakes" in df.columns
        and "area_sq_km" in df.columns
    ):

        area = df["area_sq_km"].replace(
            0, np.nan
        )

        df["lakes_per_sq_km"] = (
            df["lakes"] / area
        )

    # Households per square kilometre.
    if (
        "households" in df.columns
        and "area_sq_km" in df.columns
    ):

        area = df["area_sq_km"].replace(
            0, np.nan
        )

        df["household_density"] = (
            df["households"] / area
        )

    return df


# --------------------------------------------------
# 5. GEOGRAPHICAL FEATURES
# --------------------------------------------------

def add_geographical_features(df):

    df = df.copy()

    # Standardize coordinate names if required.
    if (
        "latitude" not in df.columns
        and "centroid_lat" in df.columns
    ):

        df["latitude"] = df["centroid_lat"]

    if (
        "longitude" not in df.columns
        and "centroid_lon" in df.columns
    ):

        df["longitude"] = df["centroid_lon"]

    # Keep coordinates numeric.
    for col in ["latitude", "longitude"]:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


# --------------------------------------------------
# 6. COMPLETE FEATURE ENGINEERING
# --------------------------------------------------

def engineer_features(df):

    df = df.copy()

    df = add_calendar_features(df)

    df = add_infrastructure_features(df)

    df = add_geographical_features(df)

    # Replace infinite values.
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df


# --------------------------------------------------
# 7. CREATE PREDICTION DATASET
# --------------------------------------------------

def create_prediction_grid(
    prediction_data,
    training_data,
    months=None,
    categories=None
):

    """
    Create ward-month-category combinations
    for zero-history wards.

    By default, uses months and categories
    observed in the training dataset.
    """

    prediction_data = prediction_data.copy()

    if months is None:

        if "month" in training_data.columns:

            months = sorted(
                training_data["month"]
                .dropna()
                .unique()
            )

        else:

            months = ["all"]

    if categories is None:

        if "category" in training_data.columns:

            categories = sorted(
                training_data["category"]
                .dropna()
                .unique()
            )

        else:

            categories = ["all"]

    if not months:
        months = ["all"]

    if not categories:
        categories = ["all"]

    # Create all combinations.
    grid = pd.MultiIndex.from_product(
        [
            prediction_data.index,
            months,
            categories
        ],
        names=[
            "ward_index",
            "month",
            "category"
        ]
    ).to_frame(index=False)

    # Attach ward infrastructure information.
    grid = grid.merge(
        prediction_data.reset_index().rename(
            columns={"index": "ward_index"}
        ),
        on="ward_index",
        how="left"
    )

    grid = grid.drop(
        columns=["ward_index"]
    )

    return grid


# --------------------------------------------------
# 8. SELECT MODEL INPUT FEATURES
# --------------------------------------------------

def select_features(df):

    df = df.copy()

    # Remove target and identifier columns.
    feature_df = df.drop(
        columns=[
            col for col in EXCLUDED_COLUMNS
            if col in df.columns
        ],
        errors="ignore"
    )

    # Keep numeric columns.
    numeric_features = feature_df.select_dtypes(
        include=np.number
    )

    # Convert remaining categorical columns
    # into one-hot encoded columns.
    categorical_features = feature_df.select_dtypes(
        exclude=np.number
    )

    categorical_features = pd.get_dummies(
        categorical_features,
        dummy_na=True,
        dtype=int
    )

    features = pd.concat(
        [
            numeric_features,
            categorical_features
        ],
        axis=1
    )

    return features


# --------------------------------------------------
# 9. PREPARE TRAINING AND PREDICTION MATRICES
# --------------------------------------------------

def build_feature_matrices(
    training_data,
    prediction_data,
    months=None,
    categories=None
):

    """
    Returns:

    X_train       -> Training input features
    y_train       -> Actual complaint counts
    X_predict     -> Zero-history ward features
    prediction_rows -> Ward IDs, months and categories
    """

    training_data = training_data.copy()

    prediction_data = prediction_data.copy()

    if TARGET_COLUMN not in training_data.columns:

        raise ValueError(
            "Training data must contain complaint_count."
        )

    # Create prediction combinations.
    prediction_rows = create_prediction_grid(
        prediction_data,
        training_data,
        months,
        categories
    )

    # Apply feature engineering.
    train_engineered = engineer_features(
        training_data
    )

    predict_engineered = engineer_features(
        prediction_rows
    )

    # Select target.
    y_train = pd.to_numeric(
        train_engineered[TARGET_COLUMN],
        errors="coerce"
    )

    valid_rows = y_train.notna() & (
        y_train >= 0
    )

    train_engineered = train_engineered.loc[
        valid_rows
    ].copy()

    y_train = y_train.loc[
        valid_rows
    ].astype(float)

    if train_engineered.empty:

        raise ValueError(
            "No valid training rows found."
        )

    # Remove identifiers and target.
    X_train = select_features(
        train_engineered
    )

    X_predict = select_features(
        predict_engineered
    )

    # Align columns between both datasets.
    X_train, X_predict = X_train.align(
        X_predict,
        join="outer",
        axis=1,
        fill_value=0
    )

    # Convert all features to numeric.
    X_train = X_train.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_predict = X_predict.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Replace invalid values.
    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_predict = X_predict.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Impute using training medians only.
    medians = X_train.median().fillna(0)

    X_train = X_train.fillna(medians)

    X_predict = X_predict.fillna(medians)

    # Remove constant features.
    useful_columns = X_train.columns[
        X_train.nunique(dropna=False) > 1
    ]

    X_train = X_train[useful_columns]

    X_predict = X_predict[useful_columns]

    return (
        X_train,
        y_train,
        X_predict,
        prediction_rows
    )


# --------------------------------------------------
# 10. TEST MODULE
# --------------------------------------------------

if __name__ == "__main__":

    from pathlib import Path

    from .preprocess import preprocess_datasets

    BASE_DIR = Path(__file__).resolve().parents[2]

    ward_file = (
        BASE_DIR / "data" / "ward_features.csv"
    )

    complaint_file = (
        BASE_DIR / "data" / "historical_complaints.csv"
    )

    training_data, prediction_data = (
        preprocess_datasets(
            ward_file,
            complaint_file
        )
    )

    X_train, y_train, X_predict, prediction_rows = (
        build_feature_matrices(
            training_data,
            prediction_data
        )
    )

    print("\n========== FEATURE ENGINEERING REPORT ==========")

    print("Training features:", X_train.shape)

    print("Training targets:", y_train.shape)

    print("Prediction features:", X_predict.shape)

    print("\nGenerated features:")

    print(X_train.columns.tolist())

    print("\nTraining feature preview:")

    print(X_train.head())

    print("\n================================================")