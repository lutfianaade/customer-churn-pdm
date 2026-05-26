# src/app.py

import os
import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import joblib
from fastapi.responses import HTMLResponse
from fastapi import Form

# ================= LOAD ENV =================
# Load .env dari root project (satu folder di atas src/)
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# ================= KONFIGURASI =================
MLFLOW_TRACKING_URI      = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")
MODEL_NAME               = os.getenv("MODEL_NAME", "ChurnModel")
MODEL_ALIAS              = os.getenv("MODEL_ALIAS", "champion")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ================= LOAD MODEL DARI MLFLOW REGISTRY =================
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "best_model.pkl"

print("MODEL PATH:", model_path)

model = joblib.load(model_path)

# ================= APP =================
app = FastAPI(title="Customer Churn Prediction")

# ================= INPUT SCHEMA =================
class CustomerData(BaseModel):
    SeniorCitizen: int
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    gender_Male: int
    Partner_Yes: int
    Dependents_Yes: int
    PhoneService_Yes: int
    MultipleLines_No_phone_service: int
    MultipleLines_Yes: int
    InternetService_Fiber_optic: int
    InternetService_No: int
    OnlineSecurity_No_internet_service: int
    OnlineSecurity_Yes: int
    OnlineBackup_No_internet_service: int
    OnlineBackup_Yes: int
    DeviceProtection_No_internet_service: int
    DeviceProtection_Yes: int
    TechSupport_No_internet_service: int
    TechSupport_Yes: int
    StreamingTV_No_internet_service: int
    StreamingTV_Yes: int
    StreamingMovies_No_internet_service: int
    StreamingMovies_Yes: int
    Contract_One_year: int
    Contract_Two_year: int
    PaperlessBilling_Yes: int
    PaymentMethod_Credit_card_automatic: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int

# ================= ENDPOINT =================
@app.get("/", response_class=HTMLResponse)
def home():

    return """

    <html>

    <head>

        <title>Customer Churn Prediction</title>

        <style>

            body{
                font-family: Arial;
                background:#f4f4f4;
                padding:40px;
            }

            .container{
                width:500px;
                margin:auto;
                background:white;
                padding:30px;
                border-radius:15px;
                box-shadow:0 0 10px rgba(0,0,0,0.1);
            }

            h1{
                text-align:center;
                color:#333;
            }

            label{
                font-weight:bold;
            }

            input, select{
                width:100%;
                padding:10px;
                margin-top:5px;
                margin-bottom:20px;
                border-radius:8px;
                border:1px solid #ccc;
            }

            button{
                width:100%;
                padding:12px;
                background:#4CAF50;
                color:white;
                border:none;
                border-radius:8px;
                font-size:16px;
                cursor:pointer;
            }

            button:hover{
                background:#45a049;
            }

        </style>

    </head>

    <body>

        <div class="container">

            <h1>Customer Churn Prediction</h1>

            <form action="/predict_form" method="post">

                <label>Tenure</label>
                <input type="number" name="tenure" required>

                <label>Monthly Charges</label>
                <input type="number" step="any" name="MonthlyCharges" required>

                <label>Total Charges</label>
                <input type="number" step="any" name="TotalCharges" required>

                <label>Contract</label>
                <select name="Contract">
                    <option value="Month-to-month">Month-to-month</option>
                    <option value="One year">One year</option>
                    <option value="Two year">Two year</option>
                </select>

                <label>Payment Method</label>
                <select name="PaymentMethod">
                    <option value="Electronic check">Electronic check</option>
                    <option value="Mailed check">Mailed check</option>
                    <option value="Credit card (automatic)">Credit Card</option>
                </select>

                <button type="submit">
                    Predict
                </button>

            </form>

        </div>

    </body>

    </html>

    """
@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(

    tenure: float = Form(...),
    MonthlyCharges: float = Form(...),
    TotalCharges: float = Form(...),
    Contract: str = Form(...),
    PaymentMethod: str = Form(...),
):

    input_dict = {

        'SeniorCitizen': 0,
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,

        'gender_Male': 1,
        'Partner_Yes': 0,
        'Dependents_Yes': 0,
        'PhoneService_Yes': 1,

        'MultipleLines_No phone service': 0,
        'MultipleLines_Yes': 0,

        'InternetService_Fiber optic': 0,
        'InternetService_No': 0,

        'OnlineSecurity_No internet service': 0,
        'OnlineSecurity_Yes': 0,

        'OnlineBackup_No internet service': 0,
        'OnlineBackup_Yes': 0,

        'DeviceProtection_No internet service': 0,
        'DeviceProtection_Yes': 0,

        'TechSupport_No internet service': 0,
        'TechSupport_Yes': 0,

        'StreamingTV_No internet service': 0,
        'StreamingTV_Yes': 0,

        'StreamingMovies_No internet service': 0,
        'StreamingMovies_Yes': 0,

        'Contract_One year': 1 if Contract == "One year" else 0,
        'Contract_Two year': 1 if Contract == "Two year" else 0,

        'PaperlessBilling_Yes': 0,

        'PaymentMethod_Credit card (automatic)': 1 if PaymentMethod == "Credit card (automatic)" else 0,

        'PaymentMethod_Electronic check': 1 if PaymentMethod == "Electronic check" else 0,

        'PaymentMethod_Mailed check': 1 if PaymentMethod == "Mailed check" else 0,
    }

    df_input = pd.DataFrame([input_dict])

    prediction = model.predict(df_input)[0]

    result = "Customer Berpotensi Churn" if prediction == 1 else "Customer Tidak Churn"

    return f"""

    <html>

    <head>

        <title>Hasil Prediksi</title>

        <style>

            body{{
                font-family:Arial;
                background:#f4f4f4;
                padding:50px;
            }}

            .box{{
                width:500px;
                margin:auto;
                background:white;
                padding:30px;
                border-radius:15px;
                text-align:center;
                box-shadow:0 0 10px rgba(0,0,0,0.1);
            }}

            h1{{
                color:#333;
            }}

            .result{{
                font-size:28px;
                color:green;
                margin-top:20px;
                font-weight:bold;
            }}

            a{{
                display:inline-block;
                margin-top:30px;
                text-decoration:none;
                background:#4CAF50;
                color:white;
                padding:12px 20px;
                border-radius:8px;
            }}

        </style>

    </head>

    <body>

        <div class="box">

            <h1>Hasil Prediksi</h1>

            <div class="result">
                {result}
            </div>

            <a href="/">
                Kembali
            </a>

        </div>

    </body>

    </html>

    """
@app.post("/predict")
def predict(data: CustomerData):
    input_dict = {
        'SeniorCitizen'                         : data.SeniorCitizen,
        'tenure'                                : data.tenure,
        'MonthlyCharges'                        : data.MonthlyCharges,
        'TotalCharges'                          : data.TotalCharges,
        'gender_Male'                           : data.gender_Male,
        'Partner_Yes'                           : data.Partner_Yes,
        'Dependents_Yes'                        : data.Dependents_Yes,
        'PhoneService_Yes'                      : data.PhoneService_Yes,
        'MultipleLines_No phone service'        : data.MultipleLines_No_phone_service,
        'MultipleLines_Yes'                     : data.MultipleLines_Yes,
        'InternetService_Fiber optic'           : data.InternetService_Fiber_optic,
        'InternetService_No'                    : data.InternetService_No,
        'OnlineSecurity_No internet service'    : data.OnlineSecurity_No_internet_service,
        'OnlineSecurity_Yes'                    : data.OnlineSecurity_Yes,
        'OnlineBackup_No internet service'      : data.OnlineBackup_No_internet_service,
        'OnlineBackup_Yes'                      : data.OnlineBackup_Yes,
        'DeviceProtection_No internet service'  : data.DeviceProtection_No_internet_service,
        'DeviceProtection_Yes'                  : data.DeviceProtection_Yes,
        'TechSupport_No internet service'       : data.TechSupport_No_internet_service,
        'TechSupport_Yes'                       : data.TechSupport_Yes,
        'StreamingTV_No internet service'       : data.StreamingTV_No_internet_service,
        'StreamingTV_Yes'                       : data.StreamingTV_Yes,
        'StreamingMovies_No internet service'   : data.StreamingMovies_No_internet_service,
        'StreamingMovies_Yes'                   : data.StreamingMovies_Yes,
        'Contract_One year'                     : data.Contract_One_year,
        'Contract_Two year'                     : data.Contract_Two_year,
        'PaperlessBilling_Yes'                  : data.PaperlessBilling_Yes,
        'PaymentMethod_Credit card (automatic)' : data.PaymentMethod_Credit_card_automatic,
        'PaymentMethod_Electronic check'        : data.PaymentMethod_Electronic_check,
        'PaymentMethod_Mailed check'            : data.PaymentMethod_Mailed_check,
    }

    df_input   = pd.DataFrame([input_dict])
    prediction = model.predict(df_input)[0]

    return {
        "Hasil Prediksi":
        "Customer Berpotensi Churn"
        if prediction == 1
        else
        "Customer Tidak Churn"
    }