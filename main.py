from fastapi import FastAPI, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
from preprocessing import preprocess_data
import joblib
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR/"xgb_model.pkl")
freq_maps = joblib.load(BASE_DIR/"freq_maps.pkl")
agg_maps = joblib.load(BASE_DIR/"agg_maps.pkl")
l_enc = joblib.load(BASE_DIR/"encoders.pkl")
feature_colum = joblib.load(BASE_DIR/"feature_colum.pkl")
feature_dtype = joblib.load(BASE_DIR/"feature_dtype.pkl")


app = FastAPI(
   title="Timo's IEEE-CIS Fraud Detection API",
   description="""
   ### Fraud Detection Service

   This API uses a trained XGBoost machine learning model

   to identify potentially fraudulent financial transactions.

   ### Features 

   -Transaction preprocessing

   -Fraud probability prediction

   -Fraudulent and legitimate classification

   -Input validation and error handling

   ### Prediction labels

   - 0: Legitimate transaction

   - 1: Fraudulent transaction

   ### Main endpoint

   - POST /predict: Submit transaction data for fraud prediction.""",

   version="1.0.0",)

app.mount("/static", 
StaticFiles(directory="frontend"), 
name="static")

@app.get("/")

def home():

    return FileResponse("frontend/index.html")

@app.post("/predict",

summary="predict transaction fraud",

description="""
submit a financial transaction to determine whether it is potentially

fraudulent or legitimate.

The API preprocesses the transaction and passes it

to the trained XGBoost model for prediction.


Returns

- prediction: 0 or 1

- result: Legitimate or Frauduent

- probability: predicted probability""")

def predict(transaction: dict = Body(...)):
    if not transaction:
       raise HTTPException(status_code=400, detail="Transaction data cannot be empty.")
    required_cols = (set(freq_maps.keys())| set(l_enc.keys())| {'card1', 'addr1', 'P_emaildomain'})
    for main_column, group_column, agg_type in agg_maps.keys():
         required_cols.add(main_column)
         required_cols.add(group_column)
    generated_cols = {'card1_addr1', 'card1_addr1_P_emaildomain'}
    required_cols -= generated_cols
    missing_cols = required_cols - set(transaction.keys())
    if missing_cols:
         raise HTTPException(status_code=400, detail=f"Missing required fields: {sorted(missing_cols)}")
    try:
        df = pd.DataFrame([transaction])
        num_cols = [col for col, dtype in feature_dtype.items() if dtype == 'float64' and col in df.columns]
        df[num_cols] = df[num_cols].astype('float64')

        X = preprocess_data(df, freq_maps, agg_maps, l_enc, feature_colum)
        print("MISSING:", set(feature_colum)-set(X.columns))
        print("Duplicate columns:", X.columns.duplicated().sum())
        if X.shape[1] != len(feature_colum):
           raise ValueError(
              f"Incorrect number of features. "
              f"Received: {X.shape[1]}, "
              f"Expected: {len(feature_colum)}")

        probability = float(model.predict_proba(X)[0, 1])
        prediction = int(probability>=0.5)
        return {"fraud_probability": probability, "prediction": prediction, "result": "fraudulent" if prediction ==1 else "legitimate"}
    except HTTPException:
        raise
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid transaction data: {str(e)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

