import asyncio
import os
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from functools import wraps
from threading import Thread
from services.telegram_bot.bot import start_polling, stop_bot, run_bot
from services.fastapi_endpoint.api_manager import start_api, stop_api, run_api
from services.flask_admin.chat_manager import start_chat, stop_chat, run_chat
import logging
import numpy as np
from models.model_loader import (
    vectorizer,
    logistic_regression_model,
    svm_model,
    random_forest_model,
    CNN_model,
    text_preprocessor
)


admin_bp = Blueprint('admin', __name__, 
                     template_folder='templates',
                     static_folder='static')

ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'password'

def preprocess_and_vectorize(text):
    try:
        preprocessed_text = text_preprocessor.preprocess_text(text)
        if not preprocessed_text:
            raise ValueError("Предобработанный текст пуст.")
        
        vectorized_text = vectorizer.transform([preprocessed_text])
        if vectorized_text.shape[1] != 1316:
            raise ValueError("Неверная размерность векторизованного текста")
            
        return vectorized_text.toarray()
    except Exception as e:
        logging.error(f"Ошибка предобработки: {e}")
        raise

def process_user_message(message, command='predict'):
    try:
        dense_vector = preprocess_and_vectorize(message)
        
        if command == 'all_predict':
            results = {
                'logistic_regression': logistic_regression_model.predict(dense_vector)[0],
                'svm': svm_model.predict(dense_vector)[0],
                'random_forest': random_forest_model.predict(dense_vector)[0],
                'cnn': CNN_model.predict(dense_vector.astype(np.float32))[0]
            }
            return format_all_predictions(results)
        
        elif command == 'logistic_regression_predict':
            return f"Logistic Regression prediction: {logistic_regression_model.predict(dense_vector)[0]}"
        
        elif command == 'svm_predict':
            return f"SVM prediction: {svm_model.predict(dense_vector)[0]}"
        
        elif command == 'random_forest_predict':
            return f"Random Forest prediction: {random_forest_model.predict(dense_vector)[0]}"
        
        elif command == 'cnn_predict':
            return f"TextCNN prediction: {CNN_model.predict(dense_vector.astype(np.float32))[0]}"
        
        elif command == 'predict':
            lr = logistic_regression_model.predict(dense_vector)[0]
            svm = svm_model.predict(dense_vector)[0]
            rf = random_forest_model.predict(dense_vector)[0]
            cnn = CNN_model.predict(dense_vector.astype(np.float32))[0]
            avg = (lr + svm + rf + cnn) / 4
            return format_main_prediction(avg)
        
        else:
            return "Неизвестная команда. Пожалуйста, выберите одну из доступных команд."
            
    except Exception as e:
        logging.error(f"Ошибка предсказания: {e}")
        return "Ошибка при обработке запроса"

def format_main_prediction(avg):
    probability = avg * 100
    if avg > 0.5:
        return f'Предположительно молния с вероятностью {probability:.2f}%'
    return f'Не молния. Вероятность {probability:.2f}% слишком мала'

def format_all_predictions(results):
    return (
        "<b>Результаты всех моделей:</b><br>"
        f"Logistic Regression: {results['logistic_regression']}<br>"
        f"SVM: {results['svm']}<br>"
        f"Random Forest: {results['random_forest']}<br>"
        f"CNN: {results['cnn']}"
    )

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session:
            return redirect(url_for('admin.login'))
        return f(*args, **kwargs)
    return decorated_function

@admin_bp.route('/chat', methods=['POST'])
@login_required
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        command = data.get('command', 'predict')

        if not user_message:
            return jsonify({"error": "Введите текст обращения"}), 400

        response = process_user_message(user_message, command)
        return jsonify({"response": response})

    except Exception as e:
        logging.error(f"Ошибка чата: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session:
            return redirect(url_for('admin.login'))
        return f(*args, **kwargs)
    return decorated_function

bot_thread = None
bot_active = False

@admin_bp.route('/admin/toggle_bot', methods=['POST'])
@login_required
def toggle_bot():
    global bot_thread, bot_active
    action = request.form.get('action')
    
    if action == 'start' and not bot_active:
        try:
            bot_thread = Thread(target=run_bot)
            bot_thread.start()
            bot_active = True
            flash('Бот успешно запущен', 'success')
        except Exception as e:
            flash(f'Ошибка при запуске бота: {str(e)}', 'error')
    
    elif action == 'stop' and bot_active:
        try:
            asyncio.run(stop_bot())
            if bot_thread and bot_thread.is_alive():
                bot_thread.join(timeout=5)
            bot_active = False
            flash('Бот успешно остановлен', 'success')
        except Exception as e:
            flash(f'Ошибка при остановке бота: {str(e)}', 'error')
    
    return redirect(url_for('admin.index'))

api_thread = None
api_active = False

@admin_bp.route('/admin/toggle_api', methods=['POST'])
@login_required
def toggle_api():
    global api_thread, api_active
    action = request.form.get('action')
    
    if action == 'start' and not api_active:
        try:
            api_thread = Thread(target=run_api)
            api_thread.start()
            api_active = True
            flash('API успешно запущено', 'success')
        except Exception as e:
            flash(f'Ошибка при запуске API: {str(e)}', 'error')
    
    elif action == 'stop' and api_active:
        try:
            asyncio.run(stop_api())
            if api_thread and api_thread.is_alive():
                api_thread.join(timeout=5)
            api_active = False
            flash('API успешно остановлено', 'success')
        except Exception as e:
            flash(f'Ошибка при остановке API: {str(e)}', 'error')
    
    return redirect(url_for('admin.index'))

chat_thread = None
chat_active = False

@admin_bp.route('/admin/toggle_chat', methods=['POST'])
@login_required
def toggle_chat():
    global chat_thread, chat_active
    action = request.form.get('action')
    
    if action == 'start' and not chat_active:
        try:
            chat_thread = Thread(target=run_chat)
            chat_thread.start()
            chat_active = True
            flash('Чат успешно запущен', 'success')
        except Exception as e:
            flash(f'Ошибка при запуске чата: {str(e)}', 'error')
    
    elif action == 'stop' and chat_active:
        try:
            asyncio.run(stop_chat())
            if chat_thread and chat_thread.is_alive():
                chat_thread.join(timeout=5)
            chat_active = False
            flash('Чат успешно остановлен', 'success')
        except Exception as e:
            flash(f'Ошибка при остановке чата: {str(e)}', 'error')
    
    return redirect(url_for('admin.index'))

@admin_bp.route('/')
@login_required
def index():
    global bot_active
    global api_active
    global chat_active

    return render_template('index.html', 
                           bot_active=bot_active,
                           api_active=api_active,
                           chat_active=chat_active)
                           

@admin_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            return redirect(url_for('admin.index'))
        else:
            flash('Неверное имя пользователя или пароль', 'error')
    
    return render_template('login.html')

@admin_bp.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin.login'))
