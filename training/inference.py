import mlflow.sklearn
import pandas as pd
import os

def model_fn(model_dir):
    return mlflow.sklearn.load_model(model_dir)

def predict_fn(input_data,model):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    
    species = {0: "setosa",1:"versicolor",2:"virginica"}
    return {"species":species[prediction]}