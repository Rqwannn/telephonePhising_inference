from flask import request, abort, jsonify
from flask_restful import Resource, reqparse
from werkzeug.datastructures import FileStorage
from sklearn.preprocessing import MultiLabelBinarizer

from df.enhance import init_df
from huggingface_hub import HfApi, upload_folder
from utils.tokenizer import *
from utils.transkrip import *
from utils.noise_removal import *

import time
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
        self.parser.add_argument('denoised', type=int, location='form', required=True)

        self.ALLOWED_EXTENSIONS = {'.mp3', '.wav'}
        self.labels = ['1_p_p_o', '2_p_b_k_k', '3_p_i_i', '4_p_j_b_j', '5_p_h', '6_n_p']

        mlb = MultiLabelBinarizer()
        processed_labels = [label.split(',') for label in self.labels]
        mlb.fit(processed_labels)

        self.model_denoised, self.df_state, _ = init_df()

        self.mlb = mlb

    def post(self):
        try:
            start_time = time.time()

            args = self.parser.parse_args()
            file = args['audio']
            denoised = args['denoised']

            tokenizer, model = load_model_from_huggingface()
            
            if denoised == 1:
                processed_data = process_audio_files(file, self.model_denoised, self.df_state)
            else:
                processed_data = process_audio_files(file)
        
            transcription_data = process_and_transcribe_audio_with_diarization(processed_data)

            input_ids, attention_masks = tokenize_with_special_tokens_and_overlap(
                transcription_data[0]["transcription_segments"],
                tokenizer
            )

            # chunk_outputs = predict(input_ids, attention_masks, model)

            chunk_outputs = []

            input_ids = torch.tensor([input_ids])
            attention_masks = torch.tensor([attention_masks])

            with torch.no_grad():
                for i in range(input_ids.size(1)):
                    chunk_input_ids = input_ids[:, i, :]
                    chunk_attention_mask = attention_masks[:, i, :]

                    chunk_output = model(
                        input_ids=chunk_input_ids,
                        attention_mask=chunk_attention_mask
                    )

                    chunk_outputs.append(chunk_output.logits.unsqueeze(1))

            logits = torch.cat(chunk_outputs, dim=1)
            weights = torch.softmax(torch.tensor([1.0] * input_ids.size(1)).to(logits.device), dim=0)
            logits = (logits * weights.unsqueeze(-1)).sum(dim=1)

            probabilities = torch.sigmoid(logits)

            threshold = 0.0
            predicted_indices = torch.where(probabilities > threshold)[1]

            predicted_labels = self.mlb.classes_[predicted_indices.cpu().numpy()]
            confidence_scores = probabilities[0, predicted_indices].cpu().numpy()

            predicted_labels_str = [str(label) for label in predicted_labels]

            end_time = time.time()  
            total_time = end_time - start_time 

            return {
                "message": "Prediction successful",
                "confidence_scores": [f'{conf * 100:.2f}%' for conf in confidence_scores],
                "predicted_labels": predicted_labels_str,
                "transcription": transcription_data[0]["transcription_segments"],
                "processing_time": f"{total_time:.4f} seconds"
            }, 200

        except Exception as e:
            return {"message": f"Error during prediction: {str(e)}"}, 500

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