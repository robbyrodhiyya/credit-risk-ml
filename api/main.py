import pandas as pd
import joblib

from fastapi import FastAPI

app = FastAPI()

# load model dan feature list
model = joblib.load("model/credit_model.pkl")
feature_columns = joblib.load("model/features.pkl")


@app.get("/")
def home():
    return {"message": "Credit Risk Prediction API"}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    # lakukan encoding seperti saat training
    df = pd.get_dummies(df)

    # pastikan semua kolom training ada
    df = df.reindex(columns=feature_columns, fill_value=0)

    # prediksi probability default
    prob = model.predict_proba(df)[0][1]

    return {
        "default_probability": float(prob)
    }