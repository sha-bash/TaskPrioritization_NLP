# Сервис для определения приоритета обращений

Данный проект предоставляет набор инструментов для предсказания приоритетов обращений с использованием моделей машинного обучения. Он включает:
- Telegram-бот для взаимодействия с пользователем.
- API на FastAPI для интеграции с другими сервисами.

## Функционал Telegram-бота

Telegram-бот предоставляет обработку различных команд для получения предсказаний от моделей. Список поддерживаемых команд:

### Команды бота

1. **`/all_predict`**: возвращает предсказания всех моделей, включая:
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Random Forest
   - TextCNN

2. **`/logistic_regression_predict`**: возвращает предсказание модели Logistic Regression.

3. **`/svm_predict`**: возвращает предсказание модели SVM.

4. **`/random_forest_predict`**: возвращает предсказание модели Random Forest.

5. **`/cnn_predict`**: возвращает предсказание модели TextCNN.

6. **`/predict`**: возвращает усреднённое предсказание всех моделей с оценкой вероятности.

7. **`/bert_predict`**: возвращает предсказание модели BERT с вероятностью.

### Логика работы
1. Пользователь вводит одну из доступных команд.
2. Бот запрашивает текст обращения.
3. После получения текста бот обрабатывает его и возвращает предсказание в зависимости от выбранной команды.

## API на FastAPI

FastAPI предоставляет эндпоинты для получения предсказаний от моделей. Все эндпоинты поддерживают обработку текста в формате JSON.

### Эндпоинты

1. **`GET /status`**
   - **Описание**: проверка статуса API.
   - **Ответ**: `{ "status": "API is running" }`

2. **`POST /v1/all_predict`**
   - **Описание**: возвращает предсказания всех моделей (Logistic Regression, SVM, Random Forest, TextCNN).
   - **Тело запроса**:
     ```json
     {
       "text": "Ваш текст для анализа"
     }
     ```
   - **Ответ**:
     ```json
     {
       "logistic_regression_prediction": 0,
       "svm_prediction": 1,
       "random_forest_prediction": 0,
       "CNN_prediction": 0.8
     }
     ```

3. **`POST /v1/logistic_regression_prediction`**
   - **Описание**: предсказание модели Logistic Regression.
   - **Тело запроса**: аналогично `all_predict`.
   - **Ответ**:
     ```json
     {
       "logistic_regression_prediction": 1
     }
     ```

4. **`POST /v1/svm_prediction`**
   - **Описание**: предсказание модели SVM.
   - **Ответ**:
     ```json
     {
       "svm_prediction": 0
     }
     ```

5. **`POST /v1/random_forest_prediction`**
   - **Описание**: предсказание модели Random Forest.
   - **Ответ**:
     ```json
     {
       "random_forest_prediction": 1
     }
     ```

6. **`POST /v1/predict`**
   - **Описание**: усреднённое предсказание всех моделей.
   - **Ответ**:
     ```json
     {
       "predict": "Предположительно молния с вероятностью 70.00%"
     }
     ```

7. **`POST /v1/CNN_prediction`**
   - **Описание**: предсказание модели TextCNN.
   - **Ответ**:
     ```json
     {
       "CNN_prediction": 0.65
     }
     ```

8. **`POST /v1/bert_prediction`**
   - **Описание**: предсказание модели BERT.
   - **Ответ**:
     ```json
     {
       "BERT_predict": "Ожидаемый результат: молния с точностью 85.00%"
     }
     ```


## Установка и запуск

### Запуск с использованием Docker

1. Постройте Docker-образ:
   ```bash
   docker build -t priority-service .
   ```

2. Запустите контейнер:
   ```bash
   docker run -d -p 8000:8000 priority-service
   ```