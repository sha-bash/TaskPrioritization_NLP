import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging


class BaseModel:
    def __init__(self, model):
        self.model = model

    def __str__(self):
        return str(self.model)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def load(self, filename):
        load_path = os.path.abspath(os.path.join('training/trained_models', filename))
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Файл модели {load_path} не найден.")
        self.model = joblib.load(load_path)
        logging.info(f'Модель загружена из {load_path}')
        return self.model

class LogisticRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(LogisticRegression(**kwargs))

class SVMModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(SVC(**kwargs))

class RandomForestModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(RandomForestClassifier(**kwargs))