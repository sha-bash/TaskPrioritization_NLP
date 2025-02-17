from aiogram import Router, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
import logging
import numpy as np

from models.model_loader import (
    vectorizer,
    logistic_regression_model,
    svm_model,
    random_forest_model,
    CNN_model,
    text_preprocessor,
    bert_nlp
)

router = Router()

logging.basicConfig(level=logging.INFO)

class EmployeeStatusForm(StatesGroup):
    waiting_for_text = State()

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


@router.message(Command("all_predict"))
@router.message(Command("logistic_regression_predict"))
@router.message(Command("svm_predict"))
@router.message(Command("random_forest_predict"))
@router.message(Command("cnn_predict"))
@router.message(Command("predict"))
@router.message(Command("bert_predict"))
async def start_status(message: types.Message, state: FSMContext):
    command = message.text.strip().lstrip("/") 
    await state.update_data(current_command=command)  
    await state.set_state(EmployeeStatusForm.waiting_for_text)
    await message.reply("Пожалуйста, введите строку обращения.")

@router.message(EmployeeStatusForm.waiting_for_text)
async def send_prediction_status(message: types.Message, state: FSMContext):
    text = message.text
    logging.info(f"Обработка текста: {text}")

    state_data = await state.get_data()
    current_command = state_data.get("current_command")

    try:
        preprocessed_text, dense_vectorized_text = preprocess_and_vectorize_text(text)

        if current_command == "all_predict":
            logistic_regression_prediction = logistic_regression_model.predict(dense_vectorized_text)[0]
            svm_prediction = svm_model.predict(dense_vectorized_text)[0]
            random_forest_prediction = random_forest_model.predict(dense_vectorized_text)[0]
            CNN_prediction = CNN_model.predict(dense_vectorized_text.astype(np.float32))

            response =  f"<b>Результаты всех моделей:</b>\n" \
                        f"logistic_regression_prediction: {logistic_regression_prediction}\n" \
                        f"svm_prediction: {svm_prediction}\n" \
                        f"random_forest_prediction: {random_forest_prediction}\n" \
                        f"CNN_prediction: {CNN_prediction}\n"
        
        elif current_command == "logistic_regression_predict":
            logistic_regression_prediction = logistic_regression_model.predict(dense_vectorized_text)[0]
            response = f"<b>Logistic Regression prediction:</b> {logistic_regression_prediction}"
        
        elif current_command == "svm_predict":
            svm_prediction = svm_model.predict(dense_vectorized_text)[0]
            response = f"<b>SVM prediction:</b> {svm_prediction}"
        
        elif current_command == "random_forest_predict":
            random_forest_prediction = random_forest_model.predict(dense_vectorized_text)[0]
            response = f"<b>Random Forest prediction:</b> {random_forest_prediction}"
        
        elif current_command == "cnn_predict":
            CNN_prediction = CNN_model.predict(dense_vectorized_text.astype(np.float32))
            response = f"<b>TextCNN prediction:</b> {CNN_prediction}"
        
        elif current_command == "predict":
            logistic_regression_prediction = logistic_regression_model.predict(dense_vectorized_text)[0]
            svm_prediction = svm_model.predict(dense_vectorized_text)[0]
            random_forest_prediction = random_forest_model.predict(dense_vectorized_text)[0]
            CNN_prediction = CNN_model.predict(dense_vectorized_text.astype(np.float32))

            average_prediction = (logistic_regression_prediction + svm_prediction + random_forest_prediction + CNN_prediction) / 4
            probability = average_prediction * 100

            if average_prediction > 0.5:
                response = f'Предположительно молния с вероятностью {probability:.2f}%'
            else:
                response = f'Не молния. Вероятность {probability:.2f}% слишком мала'

        elif current_command == "bert_predict":
            bert_prediction = bert_nlp(str(text))
            for dict_predict in bert_prediction:
                score = dict_predict['score'] * 100
                if dict_predict['label'] == 'LABEL_0':
                    text_bert_prediction = f'Ожидаемый результат: не молния с точностью {score:.2f}%'
                else:
                    text_bert_prediction = f'Ожидаемый результат: молния с точностью {score:.2f}%'

            response = text_bert_prediction

        else:
            response = "Неизвестная команда. Пожалуйста, выберите одну из доступных команд."

        await message.reply(response, parse_mode='HTML')

    except Exception as e:
        logging.error(f"Ошибка при обработке предсказания: {e}")
        await message.reply("Произошла ошибка при обработке запроса. Пожалуйста, попробуйте позже.")
    finally:
        await state.clear()
