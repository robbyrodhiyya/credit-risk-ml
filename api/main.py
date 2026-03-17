import pandas as pd
import joblib

from fastapi import FastAPI
from pydantic import BaseModel, Field
from enum import Enum

app = FastAPI()

# =========================
# Load model & features
# =========================

model = joblib.load("model/credit_model.pkl")
feature_columns = joblib.load("model/features.pkl")


# =========================
# USER-FRIENDLY ENUMS
# =========================

class CheckingAccount(str, Enum):
    LESS_THAN_0 = "LESS_THAN_0"
    BETWEEN_0_200 = "BETWEEN_0_200"
    MORE_THAN_200 = "MORE_THAN_200"
    NO_ACCOUNT = "NO_ACCOUNT"

class CreditHistory(str, Enum):
    NO_CREDIT = "NO_CREDIT"
    ALL_PAID = "ALL_PAID"
    PAID_TILL_NOW = "PAID_TILL_NOW"
    DELAYED = "DELAYED"
    CRITICAL = "CRITICAL"

class Purpose(str, Enum):
    NEW_CAR = "NEW_CAR"
    USED_CAR = "USED_CAR"
    FURNITURE = "FURNITURE"
    ELECTRONICS = "ELECTRONICS"
    EDUCATION = "EDUCATION"
    BUSINESS = "BUSINESS"

class Savings(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"
    NONE = "NONE"

class Employment(str, Enum):
    UNEMPLOYED = "UNEMPLOYED"
    LESS_1_YEAR = "LESS_1_YEAR"
    ONE_TO_FOUR = "ONE_TO_FOUR"
    FOUR_TO_SEVEN = "FOUR_TO_SEVEN"
    MORE_7 = "MORE_7"

class Housing(str, Enum):
    RENT = "RENT"
    OWN = "OWN"
    FREE = "FREE"


# =========================
# MAPPING KE DATASET ASLI
# =========================

checking_account_map = {
    "LESS_THAN_0": "A11",
    "BETWEEN_0_200": "A12",
    "MORE_THAN_200": "A13",
    "NO_ACCOUNT": "A14"
}

credit_history_map = {
    "NO_CREDIT": "A30",
    "ALL_PAID": "A31",
    "PAID_TILL_NOW": "A32",
    "DELAYED": "A33",
    "CRITICAL": "A34"
}

purpose_map = {
    "NEW_CAR": "A40",
    "USED_CAR": "A41",
    "FURNITURE": "A42",
    "ELECTRONICS": "A43",
    "EDUCATION": "A46",
    "BUSINESS": "A49"
}

savings_map = {
    "LOW": "A61",
    "MEDIUM": "A62",
    "HIGH": "A63",
    "VERY_HIGH": "A64",
    "NONE": "A65"
}

employment_map = {
    "UNEMPLOYED": "A71",
    "LESS_1_YEAR": "A72",
    "ONE_TO_FOUR": "A73",
    "FOUR_TO_SEVEN": "A74",
    "MORE_7": "A75"
}

housing_map = {
    "RENT": "A151",
    "OWN": "A152",
    "FREE": "A153"
}


# =========================
# INPUT SCHEMA
# =========================

class CreditInput(BaseModel):
    checking_account: CheckingAccount
    duration: int = Field(..., ge=4, le=72)
    credit_history: CreditHistory
    purpose: Purpose
    credit_amount: float = Field(..., ge=250, le=20000)
    savings: Savings
    employment: Employment
    installment_rate: int = Field(..., ge=1, le=4)
    age: int = Field(..., ge=18, le=75)
    housing: Housing


# =========================
# HELPER FUNCTION
# =========================

def map_input(data: dict):

    return {
        "checking_account": checking_account_map[data["checking_account"]],
        "duration": data["duration"],
        "credit_history": credit_history_map[data["credit_history"]],
        "purpose": purpose_map[data["purpose"]],
        "credit_amount": data["credit_amount"],
        "savings": savings_map[data["savings"]],
        "employment": employment_map[data["employment"]],
        "installment_rate": data["installment_rate"],
        "age": data["age"],
        "housing": housing_map[data["housing"]],
    }


# =========================
# API ENDPOINT
# =========================

@app.get("/")
def home():
    return {"message": "Credit Risk API (User Friendly Version)"}


@app.post("/predict")
def predict(data: CreditInput):

    # convert ke dict
    user_input = data.dict()

    # mapping ke kode dataset
    mapped_input = map_input(user_input)

    # dataframe
    df = pd.DataFrame([mapped_input])

    # encoding
    df = pd.get_dummies(df)

    # align dengan feature training
    df = df.reindex(columns=feature_columns, fill_value=0)

    # prediction
    prob = model.predict_proba(df)[0][1]

    return {
        "default_probability": float(prob)
    }