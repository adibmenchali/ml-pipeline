import mlflow.sklearn
import pandas as pd
import os

RUN_ID = "df8628c1ac4f4e51a1e409b84c74aab1"

model = mlflow.sklearn.load_model("/app/mlruns/1/models/m-8796c0c4a06b4f29bf133a30a8354251/artifacts")

#model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/model")

SPECIES = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

def predict(features: dict) -> str:
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    return SPECIES[prediction]