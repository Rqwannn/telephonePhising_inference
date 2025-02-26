from flask_socketio import send, emit
from utils.tokenizer import *
from utils.transkrip import *
from utils.noise_removal import *
from df.enhance import init_df

from sklearn.preprocessing import MultiLabelBinarizer

import time

values = {
    'slider1': 25,
    'slider2': 0,
}

def audio_socket_handlers(socketio_server):

    tokenizer, model = load_model_from_huggingface()
    model_denoised, df_state, _ = init_df()

    labels = ['1_p_p_o', '2_p_b_k_k', '3_p_i_i', '4_p_j_b_j', '5_p_h', '6_n_p']

    mlb = MultiLabelBinarizer()
    processed_labels = [label.split(',') for label in labels]
    mlb.fit(processed_labels)

    @socketio_server.on('connect')
    def test_connect():
        emit('after connect',  {'data':'Success'})

    @socketio_server.on('Slider value changed')
    def value_changed(message):
        values[message['who']] = message['data']
        emit('update value', values, broadcast=True)
        # emit('update value', message, broadcast=True)

    @socketio_server.on('Audio analysis')
    def value_changed(message):
        try:
            start_time = time.time()

            file = message['audio']
            denoised = message['denoised']
            text_previous = message['text_previous']

            if denoised == 1:
                processed_data = process_audio_files(file, model_denoised, df_state)
            else:
                processed_data = process_audio_files(file)

            transcription_data = process_and_transcribe_audio(processed_data)

            input_ids, attention_masks = tokenize_with_special_tokens_and_overlap(
                text_previous + "" + transcription_data[0]["transcription"],
                tokenizer
            )

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

            predicted_labels = mlb.classes_[predicted_indices.cpu().numpy()]
            confidence_scores = probabilities[0, predicted_indices].cpu().numpy()

            predicted_labels_str = [str(label) for label in predicted_labels]

            end_time = time.time()  
            total_time = end_time - start_time 

            final_data =  {
                "message": "Prediction successful",
                "confidence_scores": [f'{conf * 100:.2f}%' for conf in confidence_scores],
                "predicted_labels": predicted_labels_str,
                "transcription": transcription_data[0]["transcription"],
                "text_previous": text_previous,
                "processing_time": f"{total_time:.4f} seconds"
            }
            
            emit('update Audio analysis', final_data, broadcast=True)
        except Exception as e:
            return {"message": f"Error during prediction: {str(e)}"}, 500
