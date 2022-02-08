import numpy as np
import librosa

# Функция возвращает извлеченные из аудиофайла признаки
def get_features(x, sr):
    features = []
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(x, sr=sr))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(x, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(x, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(x))
    rms = np.mean(librosa.feature.rms(x))
    mfcc = librosa.feature.mfcc(x, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(x, sr=sr)

    features.append(spectral_bandwidth)
    features.append(spectral_centroid)
    features.append(spectral_rolloff)
    features.append(zero_crossing_rate)
    features.append(rms)
    features.extend([np.mean(i) for i in mfcc])
    features.extend([np.mean(i) for i in chroma_stft])

    return np.array(features).reshape((1, -1))
