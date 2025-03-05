from flask import Flask
from flask_restful import Api
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_talisman import Talisman
from dotenv import load_dotenv
import os

load_dotenv()

db = SQLAlchemy()
api = Api()
login_manager = LoginManager()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default_secret_key')
    app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'default_jwt_secret')

    jwt = JWTManager(app)
    db.init_app(app)
    
    from routes.path import AI_API_PATH

    AI_API_PATH()

    Talisman(app, content_security_policy=None, force_https=True, strict_transport_security=True)
    CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}}, supports_credentials=True)

    api.init_app(app)
    login_manager.init_app(app)

    from app.models.user import Users

    @login_manager.user_loader
    def load_user(user_id):      
        return Users.query.get(int(user_id))

    Migrate(app, db)

    return app