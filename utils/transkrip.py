import re
import time
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pyannote.audio import Pipeline
import os
import torchaudio.transforms as T
import torchaudio
from torchaudio.transforms import Resample

def process_audio_files(data, labels):
    processed_data = []

    waveform, sample_rate = torchaudio.load(data)

    if sample_rate != 16000:
        resample = Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resample(waveform)
        sample_rate = 16000

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    processed_data.append({
        'array': waveform.squeeze().numpy(),
        'sampling_rate': sample_rate,
        'file': data,
        'category': labels
    })

    return processed_data

def load_whisper():
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
    whisper_forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language="indonesian", task="transcribe")

    return whisper_processor, whisper_model, whisper_forced_decoder_ids

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
    start_time = time.time()

    whisper_processor, whisper_model, whisper_forced_decoder_ids = load_whisper()
    diarization_pipeline = load_diarization()
    
    transcription_data = []

    for audio_item in denoised_data:
        try:
            category = audio_item['category']
            audio_path = audio_item['file']
            audio_array = audio_item['array']
            sample_rate = audio_item['sampling_rate']

            diarization_result = diarization_pipeline(audio_path)
            
            speaker_segments = []
            transcription_segments = []
            transcription_text = []
            
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                start_frame = int(turn.start * sample_rate)
                end_frame = int(turn.end * sample_rate)
                segment_audio = torch.tensor(audio_array[start_frame:end_frame]).unsqueeze(0)

                input_features = whisper_processor(
                    segment_audio.squeeze(), sampling_rate=sample_rate, return_tensors="pt"
                ).input_features

                predicted_ids = whisper_model.generate(input_features, forced_decoder_ids=whisper_forced_decoder_ids)
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

            full_transcription = " ".join(transcription_text)

            print(full_transcription + "\n")

            print(" ".join(transcription_segments).strip() + "\n")

            transcription_data.append({
                'transcription_segments': " ".join(transcription_segments).strip(),
                'speaker_segments': speaker_segments
            })

        except Exception as e:
            print(f"Error processing file {audio_path}: {str(e)}")

    end_time = time.time()
    total_time = end_time - start_time

    print(f"{total_time}")
    
    return transcription_data, total_time