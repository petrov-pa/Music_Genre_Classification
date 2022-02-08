import numpy as np
import librosa


# Функция возвращает извлеченные из аудиофайла признаки
def get_features(y, sr):
    features = []
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
    rms = np.mean(librosa.feature.rms(y=y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

    features.append(spectral_bandwidth)
    features.append(spectral_centroid)
    features.append(spectral_rolloff)
    features.append(zero_crossing_rate)
    features.append(rms)
    features.extend([np.mean(i) for i in mfcc])
    features.extend([np.mean(i) for i in chroma_stft])

    return features
