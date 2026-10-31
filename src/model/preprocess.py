import pandas as pd
import numpy as np
import re
from pathlib import Path


# --------------------------------------------------
# 1. COLUMN NAME ALIASES
# --------------------------------------------------

COLUMN_ALIASES = {
    "ward_id": [
        "ward_id", "ward", "ward_no", "ward_number",
        "ward_code", "id"
    ],

    "ward_name": [
        "ward_name", "name", "wardname"
    ],

    "population": [
        "population", "population_2011",
        "total_population", "pop"
    ],

    "area_sq_km": [
        "area_sq_km", "area", "area_km",
        "area_kms", "ward_area"
    ],

    "households": [
        "households", "total_households",
        "households_census_2011"
    ],

    "road_length_km": [
        "road_length_km", "road_length_kms",
        "road_km", "total_road_length"
    ],

    "streetlights": [
        "streetlights", "street_lights",
        "no_of_streetlights"
    ],

    "parks": [
        "parks", "park_count", "no_of_parks"
    ],

    "lakes": [
        "lakes", "lake_count", "no_of_lakes"
    ],

    "population_density": [
        "population_density",
        "population_density_2011_persons_sq_km",
        "density"
    ],

    "latitude": [
        "latitude", "lat", "centroid_lat"
    ],

    "longitude": [
        "longitude", "lon", "lng", "centroid_lon"
    ],

    "category": [
        "category", "complaint_category",
        "complaint_type", "issue_type", "type"
    ],

    "month": [
        "month", "complaint_month"
    ],

    "date": [
        "date", "complaint_date",
        "created_at", "timestamp"
    ],

    "complaint_count": [
        "complaint_count", "total_complaints",
        "count", "number_of_complaints"
    ]
}


# --------------------------------------------------
# 2. STANDARDIZE COLUMN NAMES
# --------------------------------------------------

def clean_column_name(column):

    column = str(column).strip().lower()

    column = re.sub(r"[^a-z0-9]+", "_", column)

    return column.strip("_")


def normalize_columns(df):

    df = df.copy()

    df.columns = [
        clean_column_name(col)
        for col in df.columns
    ]

    rename_dict = {}

    for standard_name, aliases in COLUMN_ALIASES.items():

        for alias in aliases:

            alias = clean_column_name(alias)

            if alias in df.columns:

                if standard_name not in df.columns:
                    rename_dict[alias] = standard_name

                break

    df = df.rename(columns=rename_dict)

    return df


# --------------------------------------------------
# 3. LOAD CSV FILE
# --------------------------------------------------

def load_csv(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(
            f"Dataset is empty: {file_path}"
        )

    df = normalize_columns(df)

    return df


# --------------------------------------------------
# 4. CLEAN WARD FEATURES
# --------------------------------------------------

def load_ward_features(file_path):

    df = load_csv(file_path)

    if "ward_id" not in df.columns:
        raise ValueError(
            "Ward features dataset must contain ward_id."
        )

    # Remove missing IDs before converting to strings.
    df = df.dropna(subset=["ward_id"]).copy()

    df["ward_id"] = (
        df["ward_id"]
        .astype(str)
        .str.strip()
    )

    df = df[
        ~df["ward_id"].str.lower().isin(
            ["", "nan", "none", "null"]
        )
    ]

    # Convert numeric feature columns.
    excluded_columns = {
        "ward_id",
        "ward_name",
        "category",
        "month",
        "date",
        "city",
        "zone"
    }

    for col in df.columns:

        if col not in excluded_columns:

            converted = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce"
            )

            if converted.notna().mean() >= 0.7:
                df[col] = converted

    # Remove duplicate ward records.
    df = df.drop_duplicates(
        subset=["ward_id"],
        keep="first"
    )

    # Fill missing numerical features with median.
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_columns:

        median_value = df[col].median()

        if pd.notna(median_value):
            df[col] = df[col].fillna(median_value)

    return df.reset_index(drop=True)


# --------------------------------------------------
# 5. NORMALIZE MONTH INFORMATION
# --------------------------------------------------

def normalize_month(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    # Handle YYYY-MM or YYYY-MM-DD.
    if re.match(r"^\d{4}-\d{1,2}", value):

        date = pd.to_datetime(
            value,
            errors="coerce"
        )

        if pd.notna(date):
            return date.strftime("%Y-%m")

    # Handle integer month values.
    if value.isdigit():

        month_number = int(value)

        if 1 <= month_number <= 12:
            return f"month_{month_number:02d}"

    # Handle month names and other date formats.
    date = pd.to_datetime(
        value,
        errors="coerce"
    )

    if pd.notna(date):
        return date.strftime("%Y-%m")

    return value.lower()


# --------------------------------------------------
# 6. LOAD HISTORICAL COMPLAINT DATA
# --------------------------------------------------

def load_historical_complaints(file_path):

    df = load_csv(file_path)

    if "ward_id" not in df.columns:
        raise ValueError(
            "Historical complaints dataset must contain ward_id."
        )

    df = df.dropna(subset=["ward_id"]).copy()

    df["ward_id"] = (
        df["ward_id"]
        .astype(str)
        .str.strip()
    )

    df = df[
        ~df["ward_id"].str.lower().isin(
            ["", "nan", "none", "null"]
        )
    ]

    # If category is unavailable, use a general category.
    if "category" not in df.columns:

        df["category"] = "all"

    df["category"] = (
        df["category"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Extract month from date if month is unavailable.
    if "month" not in df.columns:

        if "date" in df.columns:

            dates = pd.to_datetime(
                df["date"],
                errors="coerce"
            )

            df["month"] = dates.dt.strftime("%Y-%m")

        else:

            df["month"] = "all"

    else:

        df["month"] = df["month"].apply(
            normalize_month
        )

    df["month"] = df["month"].fillna("unknown")

    # If complaint_count is absent,
    # assume every row represents one complaint.
    if "complaint_count" not in df.columns:

        df["complaint_count"] = 1

    df["complaint_count"] = pd.to_numeric(
        df["complaint_count"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["complaint_count"]
    )

    if (df["complaint_count"] < 0).any():

        raise ValueError(
            "Complaint counts cannot be negative."
        )

    # Aggregate complaints.
    grouped = (
        df.groupby(
            ["ward_id", "month", "category"],
            as_index=False
        )["complaint_count"]
        .sum()
    )

    return grouped


# --------------------------------------------------
# 7. VALIDATE BOTH DATASETS
# --------------------------------------------------

def validate_datasets(wards, complaints):

    ward_ids = set(
        wards["ward_id"].unique()
    )

    complaint_ward_ids = set(
        complaints["ward_id"].unique()
    )

    common_ids = (
        ward_ids & complaint_ward_ids
    )

    zero_history_ids = (
        ward_ids - complaint_ward_ids
    )

    if len(common_ids) < 2:

        raise ValueError(
            "Not enough matching wards with historical complaints."
        )

    if len(zero_history_ids) == 0:

        print(
            "Warning: No zero-history wards found."
        )

    print("\n========== DATASET REPORT ==========")

    print(
        "Total wards:",
        len(ward_ids)
    )

    print(
        "Wards with historical complaints:",
        len(complaint_ward_ids & ward_ids)
    )

    print(
        "Zero-history wards:",
        len(zero_history_ids)
    )

    print(
        "Historical complaint records:",
        len(complaints)
    )

    print("====================================\n")

    return zero_history_ids


# --------------------------------------------------
# 8. PREPARE TRAINING AND PREDICTION DATA
# --------------------------------------------------

def preprocess_datasets(
    ward_file,
    complaint_file
):

    # Load datasets.
    wards = load_ward_features(
        ward_file
    )

    complaints = load_historical_complaints(
        complaint_file
    )

    # Identify zero-history wards.
    zero_history_ids = validate_datasets(
        wards,
        complaints
    )

    # Keep only wards that have infrastructure data.
    complaints = complaints[
        complaints["ward_id"].isin(
            wards["ward_id"]
        )
    ]

    # Merge complaint records with ward features.
    training_data = complaints.merge(
        wards,
        on="ward_id",
        how="inner"
    )

    # Select wards without historical complaints.
    prediction_data = wards[
        wards["ward_id"].isin(
            zero_history_ids
        )
    ].copy()

    print(
        "Training dataset shape:",
        training_data.shape
    )

    print(
        "Prediction dataset shape:",
        prediction_data.shape
    )

    return training_data, prediction_data


# --------------------------------------------------
# 9. TEST THE MODULE
# --------------------------------------------------

if __name__ == "__main__":

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

    print("\nTraining Data Preview:")
    print(training_data.head())

    print("\nPrediction Data Preview:")
    print(prediction_data.head())