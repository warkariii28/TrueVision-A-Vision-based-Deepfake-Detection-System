from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
from flask_login import (
    UserMixin,
    login_user,
    LoginManager,
    current_user,
    logout_user,
    login_required,
)
from datetime import timedelta
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global extension instances
db = SQLAlchemy()
migrate = Migrate()
bcrypt = Bcrypt()
login_manager = LoginManager()
login_manager.session_protection = "strong"
login_manager.login_view = "login"
login_manager.login_message_category = "info"

def create_app():
    app = Flask(__name__, static_url_path='/static', static_folder='static')
    
    # Set the permanent session lifetime
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # User stays logged in for 7 days
    app.config['SESSION_COOKIE_SECURE'] = False

    # Safe publishable config (use env)
    app.secret_key = os.getenv('SECRET_KEY', 'change-this-in-production')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
        'DATABASE_URL',
        f"sqlite:///{os.path.join(BASE_DIR, 'database.db')}"
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Initialize extensions with app
    db.init_app(app)
    migrate.init_app(app, db)
    bcrypt.init_app(app)
    login_manager.init_app(app)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
