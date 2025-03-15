from flask_login import UserMixin
from datetime import datetime
from app import db
import uuid
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
import pytz

Base = declarative_base()


class Users(db.Model, UserMixin, Base):
    __tablename__ = 'users'

    user_id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = db.Column(db.String(36), unique=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(36))
    is_deleted = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.now(pytz.timezone('Asia/Jakarta')))

    def get_id(self):
        return str(self.user_id)