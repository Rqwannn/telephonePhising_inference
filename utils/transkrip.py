import re
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from pyannote.audio import Pipeline
import os
import torchaudio.transforms as T

def load_whisper():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    whisper_processor = WhisperProcessor.from_pretrained(os.getenv("WHISPER_ID"))
    whisper_model = WhisperForConditionalGeneration.from_pretrained(os.getenv("WHISPER_ID"))
    whisper_forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language="indonesian", task="transcribe")

    whisper_model.to(device)

    return whisper_processor, whisper_model, device, whisper_forced_decoder_ids

def load_diarization():
    pipeline = Pipeline.from_pretrained(
        os.getenv("DIARIZATION_ID"),
        use_auth_token=os.getenv("API_KEY_DIARIZATION"))

    return pipeline

def remove_repeated_text(text):
    text = re.sub(r"(.)\1{4,}", r"\1", text)
    text = re.sub(r"(.{2,}?)\1{2,}", r"\1", text)
    text = re.sub(r"\b(\w+)( \1){2,}\b", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def process_and_transcribe_audio_with_diarization(denoised_data):
    whisper_processor, whisper_model, device, whisper_forced_decoder_ids = load_whisper()
    diarization_pipeline = load_diarization()
    
    transcription_data = []

    for audio_item in denoised_data:
        try:
            audio_path = audio_item['file']
            audio_array = audio_item['array']
            sample_rate = audio_item['sampling_rate']

            diarization_input = {
                "waveform": audio_array,
                "sample_rate": sample_rate
            }

            diarization_result = diarization_pipeline(diarization_input)
            
            speaker_segments = []
            transcription_segments = []
            transcription_text = []
            
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                start_frame = int(turn.start * sample_rate)
                end_frame = int(turn.end * sample_rate)
                segment_audio = torch.tensor(audio_array.squeeze().numpy()[start_frame:end_frame]).unsqueeze(0)

                input_features = whisper_processor(
                    segment_audio.squeeze(), sampling_rate=sample_rate, return_tensors="pt"
                ).input_features.to(device)

                with torch.no_grad():
                    predicted_ids = whisper_model.generate(
                        input_features, 
                        forced_decoder_ids=whisper_forced_decoder_ids,
                        temperature=1.0
                    )
                
                segment_transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
               
                cleaned_segment_transcription = remove_repeated_text(segment_transcription)
                cleaned_segment_transcription = re.sub(r'\s+', ' ', cleaned_segment_transcription).strip()

                transcription_segments.append(f"[{speaker}] {cleaned_segment_transcription}")
                transcription_text.append(cleaned_segment_transcription)
                speaker_segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })

            transcription_data.append({
                'transcription_segments': " ".join(transcription_segments).strip(),
                'speaker_segments': speaker_segments
            })

        except Exception as e:
            print(f"Error processing file {audio_path}: {str(e)}")
    
    return transcription_data