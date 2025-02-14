import asyncio
import os
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from functools import wraps
from threading import Thread
from services.telegram_bot.bot import start_polling, stop_bot, run_bot
from services.fastapi_endpoint.api_manager import start_api, stop_api, run_api
import logging


admin_bp = Blueprint('admin', __name__, 
                     template_folder='templates',
                     static_folder='static')

ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'password'


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session:
            return redirect(url_for('admin.login'))
        return f(*args, **kwargs)
    return decorated_function



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


@admin_bp.route('/')
@login_required
def index():
    global bot_active
    global api_active

    return render_template('index.html', 
                           bot_active=bot_active,
                           api_active=api_active,)
                           

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
