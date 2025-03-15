from flask import request, jsonify
from flask_restful import Resource
from app import db
from datetime import datetime
import pytz
import datetime
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import create_access_token, get_jwt_identity, jwt_required, JWTManager

from app.models.user import Users

class Login(Resource):
    
    @jwt_required()
    def get(self):
        user_id = get_jwt_identity()
        value = Users.query.filter_by(user_id=user_id).first()
        return jsonify(
            username=value.username,
            email=value.email
        )

    def post(self):

        username = request.json.get('username', None)
        password = request.json.get('password', None)

        if not username or not password:
            return jsonify( {
                "message": "Username dan password harus diisi",
                "status": 400
            })
        
        # Proses autentikasi
        user = Users.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            token = create_access_token(identity=user.user_id, expires_delta=datetime.timedelta(days=1))
            
            return jsonify( {
                "token" : token,
                "data": {
                    "user_id": user.user_id,
                    "username": user.username,
                    "email": user.email
                },
                "status": 200
            } )

        else:
            # Jika autentikasi gagal
            return jsonify ( {
                "message": "Username atau password salah",
                "status": 401
            } )


class Register(Resource):
    def post(self):
        # fetch data
        username = request.json.get('username', None)
        password = request.json.get('password', None)
        email = request.json.get('email', None)

        # data validation
        if not username or not password or not email:
            return jsonify( {"message": "Formulir tidak boleh kosong!", "status": 401  } )
        elif len(password) <= 5:
            return jsonify( {"message": "Password harus berisi minimal 6 digit!", "status": 401  } )
        else:
            # save data to the database
            password_hash = generate_password_hash(password)
            values = Users(username=username, password=password_hash, email=email)

            # handler if username already use
            try:
                db.session.add(values) 
                db.session.commit()
                token = create_access_token(identity=values.user_id, expires_delta=datetime.timedelta(days=1))
            except IntegrityError:
                db.session.rollback()
                message = 'Username atau email sudah digunakan'
                return jsonify({
                    "message": message,
                    "status": 401              
                })

            # return json response
            return jsonify({
                "data": {
                    "user_id": values.user_id,
                    "username": values.username,
                    "email": values.email,
                    "token":token
                },
                "message": "Registrasi Berhasil",
                "status": 200
            })
        
    def delete(self, id):
        plan = Users.query.filter_by(user_id=id).first()

        plan.deleted_at=datetime.now(pytz.timezone('Asia/Jakarta'))
        db.session.commit()

        return jsonify(message="Data berhasil dihapus")