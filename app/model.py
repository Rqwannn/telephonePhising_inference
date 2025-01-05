from flask import request, abort, jsonify
from flask_restful import Resource, reqparse
from werkzeug.datastructures import FileStorage

from huggingface_hub import HfApi, upload_folder
import numpy as np

from dotenv import load_dotenv
import os

"""

1.	Penipuan Pinjaman Online Ilegal
2.	Penipuan berkedok krisis keluarga (Kecelakaan, Sakit, Narkoba, Tilang polisi)
3.	Penipuan Investasi Ilegal
4.	Penipuan Jual beli (Barang/Jasa tidak datang, barang/jasa tidak sesuai, uang tidak sampai ke penjual, dll) 
5.	Penipuan berkedok hadiah

"""

class Inference(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('audio', type=FileStorage, location='files', required=True)

        self.ALLOWED_EXTENSIONS = {'.mp3', '.wav'}
        self.labels = ['1_p_p_o', '2_p_b_k_k', '3_p_i_i', '4_p_j_b_j', '5_p_h', '6_n_p']

    def get(self):
        try:
            load_dotenv()

            model_path = "resource/models"
            tokenizer_path = "resource/tokenizer"
            repo_name = os.getenv("REPO_ID")
            hf_token = os.getenv("HF_TOKEN")

            if hf_token is None:
                return jsonify({
                    "message": "Tidak ada secret token Hugging Face"
                })
            print("Using Hugging Face token from environment variables.")

            if not os.path.exists(model_path):
                return jsonify({"message": f"Model directory '{model_path}' tidak ditemukan."})
            if not os.path.exists(tokenizer_path):
                return jsonify({"message": f"Tokenizer directory '{tokenizer_path}' tidak ditemukan."})

            api = HfApi()
            api.create_repo(repo_id=repo_name, token=hf_token, exist_ok=True)
            print(f"Repository {repo_name} created or already exists.")

            print("Uploading model to Hugging Face Hub...")
            upload_folder(
                folder_path=model_path,
                repo_id=repo_name,
                path_in_repo="models",  
                # path_in_repo="", simpan di root repo  
                token=hf_token,
                commit_message="Upload model"
            )

            print("Uploading tokenizer to Hugging Face Hub...")
            upload_folder(
                folder_path=tokenizer_path,
                repo_id=repo_name,
                path_in_repo="tokenizer",  
                token=hf_token,
                commit_message="Upload tokenizer"
            )

            return jsonify({
                "message": "Model dan tokenizer berhasil diunggah ke Hugging Face Hub"
            })

        except Exception as e:
            return jsonify({
                "message": f"Terjadi masalah saat mengunggah ke Hugging Face Hub: {str(e)}"
            })
    
    def post(self):
        pass