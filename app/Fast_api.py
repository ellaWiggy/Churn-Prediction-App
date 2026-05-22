import os
import sys
import pandas as pd
import shap

# fastAPI imports
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Custom imports
from src.utils.file_logs import get_logger
from src.utils.PKL_obj import load_object
# Initialize logger
log = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        log.info("API Startup: Loading model and artifacts...")
        
        vali_df = pd.read_csv("artifacts/app_validation.csv")

        if 'CLIENTNUM' in vali_df.columns:
            vali_df['CLIENTNUM'] = vali_df['CLIENTNUM'].astype(int)
            vali_df.set_index('CLIENTNUM', inplace=True)
        else:
            vali_df.index = vali_df.index.astype(int)

        app.state.df = vali_df

        model_pipeline = load_object("artifacts/model.pkl")
        app.state.model = model_pipeline
        app.state.classifier = model_pipeline.named_steps['model']
        app.state.preprocessor = model_pipeline[:-1]
        app.state.feature_names = model_pipeline.named_steps['selection'].get_feature_names_out()
        
        log.info(f"API Startup: All artifacts loaded. {len(vali_df)} customers in memory.")
        
    except Exception as e:
        log.error(f"Startup failed: {e}")
        raise e
    
    yield  
    log.info("API Shutdown: Cleaning up state.")
    
app = FastAPI(
    title="Credit Customer Churn Prediction API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/customer/{client_id}")
def get_customer_info(client_id: int, request: Request):
    df = getattr(request.app.state, "df", None)
    if df is None:
        raise HTTPException(status_code=500, detail="DataFrame not found in app state")
    try:
        log.info(f"Searching for Customer ID: {client_id}")
        customer_data = df.loc[client_id].to_dict()
        return {"data": customer_data, "status": "success"}
    
    except KeyError:
        log.warning(f"Customer ID {client_id} not found in validation set.")
        return {"status": "error", "message": f"Customer ID {client_id} not found"}
    
# Pydantic model for API input
class InputDataRequest(BaseModel):
    dependent_count: int
    education_level: str
    marital_status: str
    income_category: str
    card_category: str
    months_on_book: float
    total_relationship_count: float
    months_inactive_12_mon: float
    contacts_count_12_mon: float
    credit_limit: float
    total_revolving_bal: float
    total_amt_chng_q4_q1: float
    total_trans_amt: float
    total_trans_ct: float
    total_ct_chng_q4_q1: float
    avg_utilization_ratio: float


@app.post("/predict")
def get_prediction(data: InputDataRequest, request: Request):
    pipeline = request.app.state.model
    classifier = request.app.state.classifier
    preprocessor = request.app.state.preprocessor
    feature_names = request.app.state.feature_names

    log.info(f"Received prediction request for input:")
    df = pd.DataFrame([data.model_dump()])
    
    proba = pipeline.predict_proba(df)[:, 1][0]
    status = "Likely to Churn" if proba >= 0.5 else "Likely to Stay"
    log.info(f"Prediction: {status} (Churn Probability: {proba:.4f})")

    # Feature Importance (SHAP)
    input_processed = preprocessor.transform(df)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(input_processed)

    current_shap = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    total_impact = sum(abs(current_shap))

    feature_importance = dict(zip(feature_names, current_shap))
    top_factors = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    factors_output = [
        {
            "feature": f.split('__')[-1].replace('_', ' ').title(), 
            "impact": round((val / total_impact) * 100, 1) if total_impact > 0 else 0
        } for f, val in top_factors
    ]
    log.info(f"Top factors prediction: {factors_output}")

    return {
        'prediction': {
            "status": status,
            "probability": round(float(proba), 4),
            "top_factors": factors_output
        }
    }
    
@app.get("/batch_predict")
def batch_predict(request: Request):
    df = request.app.state.df
    pipeline = request.app.state.model

    log.info(f"Batch prediction started for {len(df)} customers.")
    proba = pipeline.predict_proba(df)[:, 1]
    log.info(f"Batch prediction completed for {len(df)} customers.")
    
    res = [
        {"clientnum": int(cid), "probability": round(float(p), 4)} 
        for cid, p in zip(df.index, proba)
    ]
    return {"results": res}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# Usage:
# python -m app.Fast_api