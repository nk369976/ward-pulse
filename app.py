from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from pathlib import Path
import pandas as pd

app = FastAPI(
    title="Ward Pulse API",
    description="AI-powered Civic Complaint Demand Prediction",
    version="1.0"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "ward_predictions.csv"

DASHBOARD_FILE = BASE_DIR / "dashboard.html"
INDEX_FILE = BASE_DIR / "index.html"


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------
# CATEGORY NORMALIZATION
# ------------------------------------------------

CATEGORY_ALIASES = {
    "electrical": "electrical",
    "electricity": "electrical",
    "streetlight": "electrical",

    "solid_waste": "solid_waste",
    "garbage": "solid_waste",
    "waste": "solid_waste",

    "roads": "roads",
    "road": "roads",

    "trees": "trees",
    "tree": "trees",

    "animals_health": "animals_health",
    "animals": "animals_health",
    "animal": "animals_health",

    "other": "other"
}


def normalize_category(category):

    value = str(category).strip().lower()

    return CATEGORY_ALIASES.get(value, "other")


# ------------------------------------------------
# LOAD PREDICTIONS
# ------------------------------------------------

def load_predictions():

    if not DATA_FILE.exists():

        raise HTTPException(
            status_code=404,
            detail="Prediction file not found. Run predict.py first."
        )

    df = pd.read_csv(DATA_FILE)

    required = {
        "ward_id",
        "month",
        "category",
        "predicted_complaints"
    }

    missing = required - set(df.columns)

    if missing:

        raise HTTPException(
            status_code=500,
            detail=f"Missing prediction columns: {list(missing)}"
        )

    df["ward_id"] = df["ward_id"].astype(str)

    df["category"] = df["category"].apply(
        normalize_category
    )

    df["predicted_complaints"] = pd.to_numeric(
        df["predicted_complaints"],
        errors="coerce"
    ).fillna(0).clip(lower=0)

    return df


# ------------------------------------------------
# HOME PAGE
# ------------------------------------------------

@app.get("/")
def home():

    return FileResponse(INDEX_FILE)


# ------------------------------------------------
# DASHBOARD
# ------------------------------------------------

@app.get("/dashboard")
@app.get("/dashboard.html")
def dashboard():

    return FileResponse(DASHBOARD_FILE)


# ------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------

@app.get("/api/health")
def health():

    return {
        "status": "running",
        "project": "Ward Pulse"
    }


# ------------------------------------------------
# MODEL PREDICTIONS
# ------------------------------------------------

@app.get("/api/predictions")
def get_predictions(month: str = None):

    df = load_predictions()

    if month:

        df = df[
            df["month"].astype(str) == month
        ]

    result = []

    for ward_id, group in df.groupby("ward_id"):

        counts = (
            group.groupby("category")[
                "predicted_complaints"
            ].sum()
        )

        category_counts = {
            category: round(float(count), 2)
            for category, count in counts.items()
        }

        total = sum(category_counts.values())

        result.append({

            "ward_id": ward_id,

            "predicted_total": round(total, 2),

            "counts": category_counts

        })

    return {
        "model": "Adaptive Regression",
        "month": month,
        "total_wards": len(result),
        "predictions": result
    }


# ------------------------------------------------
# MODEL INFORMATION
# ------------------------------------------------

@app.get("/api/model-info")
def model_info():

    import json

    report_file = (
        BASE_DIR / "artifacts" / "model_report.json"
    )

    if not report_file.exists():

        return {
            "status": "Model report unavailable"
        }

    with open(report_file) as file:

        return json.load(file)