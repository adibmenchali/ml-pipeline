import logging
from fastapi import FastAPI
from app.models import IrisFeatures, PredictionResponse
from app.predictor import predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="IRIS Classifier",version="1.0.0")

@app.get("/health")
async def health():
    return {"status":"ok"}

@app.post("/predict",response_model=PredictionResponse)
async def make_prediction(features: IrisFeatures):
    input_dict = {
        "sepal length (cm)": features.sepal_length_cm,
        "sepal width (cm)": features.sepal_width_cm,
        "petal length (cm)": features.petal_length_cm,
        "petal width (cm)": features.petal_width_cm,
    }
    species = predict(input_dict)
    logger.info(f"Predicted species: {species}")
    
    return PredictionResponse(species=species,features=features)