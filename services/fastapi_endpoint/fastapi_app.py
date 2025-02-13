import os
import json
import logging
import joblib
from fastapi import FastAPI, HTTPException
from preprocessing.textpreprocessor import TextPreprocessor
import shema
from lifespan import lifespan
from sklearn.feature_extraction.text import HashingVectorizer
from models.classic_ml.classic_ml import LogisticRegressionModel, SVMModel, RandomForestModel
from models.CNN.CNN import TextCNNModel
import numpy as np

logging.basicConfig(level=logging.INFO)

from models.model_loader import (
    vectorizer,
    logistic_regression_model,
    svm_model,
    random_forest_model,
    CNN_model,
    text_preprocessor
)

app = FastAPI(
    title='Сервис для определения приоритета',
    version='2.0.1',
    lifespan=lifespan,
    description='Сервис использует алгоритмы машинного обучения для формирования предсказания приоритета обращения'
)

def preprocess_and_vectorize_text(text):
    preprocessed_text = text_preprocessor.preprocess_text(text)
    if not preprocessed_text:
        raise ValueError("Предобработанный текст пуст.")
    logging.info(f"Preprocessed text: {preprocessed_text}")

    vectorized_text = vectorizer.transform([preprocessed_text])
    if vectorized_text.shape[1] != 1316:
        raise ValueError("Размерность векторизованного текста не соответствует ожидаемой.")
    logging.info(f"Vectorized text: {vectorized_text}")

    dense_vectorized_text = vectorized_text.toarray()
    return preprocessed_text, dense_vectorized_text

@app.post('/v1/all_predict', response_model=shema.CreateAllPredictResponse)
async def add_predict(predict_json: shema.CreatePredictRequest):
    try:
        text = str(predict_json.text)
        logging.info(f"Received text: {text}")

        preprocessed_text, dense_vectorized_text = preprocess_and_vectorize_text(text)

        # Предсказание от каждой модели
        logistic_regression_prediction = logistic_regression_model.predict(dense_vectorized_text)[0]
        logging.info(f"Logistic Regression prediction: {logistic_regression_prediction}")

        svm_prediction = svm_model.predict(dense_vectorized_text)[0]
        logging.info(f"SVM prediction: {svm_prediction}")

        random_forest_prediction = random_forest_model.predict(dense_vectorized_text)[0]
        logging.info(f"Random Forest prediction: {random_forest_prediction}")
        
        CNN_prediction = CNN_model.predict(dense_vectorized_text.astype(np.float32))
        logging.info(f"TextCNN prediction: {CNN_prediction}")

        return {
            "logistic_regression_prediction": logistic_regression_prediction,
            "svm_prediction": svm_prediction,
            "random_forest_prediction": random_forest_prediction,
            "CNN_prediction": CNN_prediction,
        }

    except ValueError as ve:
        logging.error(f"ValueError in /v1/predict: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"Error in /v1/predict: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    

@app.post('/v1/logistic_regression_prediction', response_model=shema.CreatelogisticregressionpredictionResponse)
async def add_predict(predict_json: shema.CreatePredictRequest):
    try:
        text = str(predict_json.text)
        logging.info(f"Received text: {text}")

        _, dense_vectorized_text = preprocess_and_vectorize_text(text)

        logistic_regression_prediction = logistic_regression_model.predict(dense_vectorized_text)[0]
        logging.info(f"Logistic Regression prediction: {logistic_regression_prediction}")

        return {
            "logistic_regression_prediction": logistic_regression_prediction,
        }

    except ValueError as ve:
        logging.error(f"ValueError in /v1/predict: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"Error in /v1/predict: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    

@app.post('/v1/svm_prediction', response_model=shema.CreateSVMpredictionResponse)
async def add_predict(predict_json: shema.CreatePredictRequest):
    try:
        text = str(predict_json.text)
        logging.info(f"Received text: {text}")

        _, dense_vectorized_text = preprocess_and_vectorize_text(text)

        svm_prediction = svm_model.predict(dense_vectorized_text)[0]
        logging.info(f"SVM prediction: {svm_prediction}")

        return {
            "svm_prediction": svm_prediction,
        }

    except ValueError as ve:
        logging.error(f"ValueError in /v1/predict: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"Error in /v1/predict: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
@app.post('/v1/random_forest_prediction', response_model=shema.CreateRandomForestpredictionResponse)
async def add_predict(predict_json: shema.CreatePredictRequest):
    try:
        text = str(predict_json.text)
        logging.info(f"Received text: {text}")

        _, dense_vectorized_text = preprocess_and_vectorize_text(text)

        random_forest_prediction = random_forest_model.predict(dense_vectorized_text)[0]
        logging.info(f"Random Forest prediction: {random_forest_prediction}")

        return {
            "random_forest_prediction": random_forest_prediction,
        }

    except ValueError as ve:
        logging.error(f"ValueError in /v1/predict: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"Error in /v1/predict: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
@app.post('/v1/predict', response_model=shema.CreatePredictResponse)
async def add_predict(predict_json: shema.CreatePredictRequest):
    try:
        text = str(predict_json.text)
        logging.info(f"Received text: {text}")

        _, dense_vectorized_text = preprocess_and_vectorize_text(text)

        # Предсказание от каждой модели
        logistic_regression_prediction = logistic_regression_model.predict(dense_vectorized_text)[0]
        logging.info(f"Logistic Regression prediction: {logistic_regression_prediction}")

        svm_prediction = svm_model.predict(dense_vectorized_text)[0]
        logging.info(f"SVM prediction: {svm_prediction}")

        random_forest_prediction = random_forest_model.predict(dense_vectorized_text)[0]
        logging.info(f"Random Forest prediction: {random_forest_prediction}")

        CNN_prediction = CNN_model.predict(dense_vectorized_text.astype(np.float32))
        logging.info(f"TextCNN prediction: {CNN_prediction}")

        # Вычисление среднего значения предсказаний
        average_prediction = (logistic_regression_prediction + svm_prediction + random_forest_prediction + CNN_prediction) / 4
        probability = average_prediction * 100

        if average_prediction > 0.5:
            return {
                "predict": f'Предположительно молния с вероятностью {probability:.2f}%'
            }
        else:
            return {
                "predict": f'Не молния. Вероятность {probability:.2f}% слишком мала'
            }

    except ValueError as ve:
        logging.error(f"ValueError in /v1/predict: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"Error in /v1/predict: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
@app.post('/v1/CNN_prediction', response_model=shema.CreateCNNPredictResponse)
async def add_predict(predict_json: shema.CreatePredictRequest):
    try:
        text = str(predict_json.text)
        logging.info(f"Received text: {text}")

        _, dense_vectorized_text = preprocess_and_vectorize_text(text)

        CNN_prediction = CNN_model.predict(dense_vectorized_text.astype(np.float32))
        logging.info(f"TextCNN prediction: {CNN_prediction}")

        return {
            "CNN_prediction": CNN_prediction,
        }

    except ValueError as ve:
        logging.error(f"ValueError in /v1/predict: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"Error in /v1/predict: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")