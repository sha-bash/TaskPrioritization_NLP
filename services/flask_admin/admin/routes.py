import asyncio
import os
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from functools import wraps
from threading import Thread
import logging
from Bot.bot import start_polling, stop_bot, run_bot

bot_thread = None
bot_active = False
admin_bp = Blueprint('admin', __name__, 
                     template_folder='templates',
                     static_folder='static')

ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'password'

def load_google_sheet_url():
    try:
        with open('Config/google_sheet_url.txt', 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        return ''

def save_google_sheet_url(url):
    with open('Config/google_sheet_url.txt', 'w', encoding='utf-8') as file:
        file.write(url)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session:
            return redirect(url_for('admin.login'))
        return f(*args, **kwargs)
    return decorated_function

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

@admin_bp.route('/')
@login_required
def index():
    global bot_active
    google_sheet_url = load_google_sheet_url()

    return render_template('index.html', 
                           google_sheet_url=google_sheet_url,
                           bot_active=bot_active)

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

@admin_bp.route('/update_sheet_url', methods=['POST'])
@login_required
def update_sheet_url():
    new_url = request.form.get('sheet_url')
    try:
        save_google_sheet_url(new_url)
        flash('Ссылка на таблицу успешно обновлена', 'success')
    except Exception as e:
        flash(f'Ошибка при обновлении ссылки: {str(e)}', 'error')
    return redirect(url_for('admin.index'))