import torchaudio
import numpy as np
from scipy import signal
from torchaudio.transforms import Resample
import torch
from df.enhance import enhance, load_audio, save_audio
from df.utils import download_file

def reduce_noise(waveform, sample_rate):

    """

    Wiener Filter
    
    Audio dengan noise kompleks dan berubah-ubah

    Rekaman suara dengan:


    Background noise tidak konstan
    Gangguan frekuensi acak
    Noise elektronik dinamis


    Contoh kasus:


    Rekaman telepon
    Audio komunikasi radio
    Sinyal sensor bergetar
    Rekaman medis
    Audio dengan interferensi
    
    """

    audio_np = waveform.numpy().flatten()
    f, t, Zxx = signal.stft(audio_np, fs=sample_rate, nperseg=1024)
    
    noise_psd = np.mean(np.abs(Zxx[:, :10])**2, axis=1)
    signal_psd = np.mean(np.abs(Zxx)**2, axis=1)
    
    wiener_filter = signal_psd / (signal_psd + noise_psd)
    
    Zxx_clean = Zxx * wiener_filter[:, np.newaxis]
    
    _, cleaned_signal = signal.istft(Zxx_clean, fs=sample_rate)

    cleaned_waveform = torch.from_numpy(cleaned_signal).float().unsqueeze(0)
    
    return cleaned_waveform

def resample_waveform(waveform, sample_rate):
    if sample_rate != 16000:
        resample = Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resample(waveform)
        sample_rate = 16000

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform, sample_rate

def load_and_process_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    waveform, sample_rate = resample_waveform(waveform, sample_rate)

    return waveform, sample_rate

def load_and_reduce_audio_noise(data, model, df_state):

    # Jika sample rate bukan 48k maka akan di resampling ke situ dulu

    audio, _ = load_audio(data, sr=df_state.sr())
    enhanced = enhance(model, df_state, audio)

    # enhanced -> Waveform
    # df_state.sr() -> sample_rate

    # save_audio("enhanced3.wav", enhanced, df_state.sr())
    
    waveform, sample_rate = resample_waveform(enhanced, df_state.sr())

    return waveform, sample_rate

def process_audio_files(data, model=None, df_state=None):
    processed_data = []

    try:
        if model is not None and df_state is not None:
            denoised_waveform, sample_rate = load_and_reduce_audio_noise(data, model, df_state)
        else:
            # Denoised in here doesn't work

            waveform, sample_rate = load_and_process_audio(data)
            denoised_waveform = reduce_noise(waveform, sample_rate)

        processed_data.append({
            'array': denoised_waveform,
            'sampling_rate': sample_rate,
            'file': data,
        })

    except Exception as e:
        print(f"Error memproses penghapusan noise")

    return processed_data