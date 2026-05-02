from pydantic import BaseModel

class IrisFeatures(BaseModel):
    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float
    
class PredictionResponse(BaseModel):
    species: str
    features: IrisFeatures
