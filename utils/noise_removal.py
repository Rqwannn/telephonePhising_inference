import torchaudio
import numpy as np
from scipy import signal
from torchaudio.transforms import Resample
import torch

def load_and_process_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)

    if sample_rate != 16000:
        resample = Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resample(waveform)
        sample_rate = 16000

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform, sample_rate

# def reduce_noise(waveform, sample_rate):
#     audio_np = waveform.numpy().flatten()

#     noise_sample = audio_np[:int(len(audio_np) * 0.1)]
#     noise_profile = np.mean(np.abs(noise_sample))

#     f, t, Zxx = signal.stft(audio_np, fs=sample_rate, nperseg=2048)
#     noise_magnitude = np.mean(np.abs(Zxx[:, :10]), axis=1)
#     clean_magnitude = np.maximum(0, np.abs(Zxx) - noise_magnitude[:, np.newaxis] * 2)
#     Zxx_clean = clean_magnitude * np.exp(1j * np.angle(Zxx))
#     _, cleaned_signal = signal.istft(Zxx_clean, fs=sample_rate)

#     cleaned_waveform = torch.from_numpy(cleaned_signal).float().unsqueeze(0)

#     return cleaned_waveform

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

# def reduce_noise(waveform, sample_rate):

#     """
#         Spectral Gating

#         Audio dengan background noise konstan:

#         Rekaman kantor
#         Audio dengan fan/AC
#         Lingkungan dengan noise mesin


#         Audio dengan noise frekuensi rendah:


#         Dengung elektronik
#         Noise listrik
#         Gemuruh mesin


#         Rekaman yang memiliki:


#         Noise statis
#         Pola frekuensi yang stabil
#         Energi spektral rendah di beberapa wilayah

#     """

#     audio_np = waveform.numpy().flatten()
#     f, t, Zxx = signal.stft(audio_np, fs=sample_rate)
    
#     spectral_energy = np.abs(Zxx)**2
#     threshold = np.percentile(spectral_energy, 10)
    
#     Zxx_clean = np.where(spectral_energy > threshold, Zxx, 0)
#     _, cleaned_signal = signal.istft(Zxx_clean, fs=sample_rate)
#     cleaned_waveform = torch.from_numpy(cleaned_signal).float().unsqueeze(0)
    
#     return cleaned_waveform

def process_audio_files(data):
    processed_data = []

    try:
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