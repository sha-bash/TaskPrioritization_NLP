from flask import Flask
from services.flask_admin.routes import admin_bp
import os
from services.services_config.config import read_yaml
from flask_session import Session 
import tracemalloc

tracemalloc.start()

config_path = 'services/services_config/config.yaml'
config = read_yaml(config_path)

app = Flask(__name__)

app.secret_key = config['conf']['secret_key']

app.config['SESSION_TYPE'] = 'filesystem'  # Хранение сессий в файловой системе
app.config['SESSION_FILE_DIR'] = os.path.join(os.getcwd(), 'flasksession')  # Папка для хранения сессий
app.config['SESSION_PERMANENT'] = False  # Сессии не будут постоянными
app.config['SESSION_USE_SIGNER'] = True  # Подписывать сессии для безопасности
app.config['SESSION_KEY_PREFIX'] = 'myapp_'  # Префикс для ключей сессий

Session(app)

# Регистрация blueprint для админ-панели
app.register_blueprint(admin_bp, url_prefix='/admin')

# Запуск приложения
if __name__ == '__main__':
    if not os.path.exists(app.config['SESSION_FILE_DIR']):
        os.makedirs(app.config['SESSION_FILE_DIR'])
    
    app.run(debug=True)