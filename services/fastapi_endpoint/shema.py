from pydantic import BaseModel


class CreatePredictRequest(BaseModel):
    text: str

class CreateAllPredictResponse(BaseModel):
    logistic_regression_prediction: int
    svm_prediction: int
    random_forest_prediction: int
    CNN_prediction: float

class CreatelogisticregressionpredictionResponse(BaseModel):
    logistic_regression_prediction: int

class CreateSVMpredictionResponse(BaseModel):
    svm_prediction: int

class CreateRandomForestpredictionResponse(BaseModel):
    random_forest_prediction: int
    
class CreatePredictResponse(BaseModel):
    predict: str

class CreateCNNPredictResponse(BaseModel):
    CNN_prediction: float