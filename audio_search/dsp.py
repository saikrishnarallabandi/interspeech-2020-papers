import numpy as np
import librosa
import scipy

sample_rate = 8000
n_fft = 2048
fft_bins = n_fft // 2 + 1
num_mels = 80
hop_length = 64
win_length = 1024
fmin = 40
min_level_db = -100
ref_level_db = 20


def load_wav(filename, encode=True):
    x = librosa.load(filename, sr=sample_rate)[0]
    if encode:
        x = encode_16bits(x)
    return x


def save_wav(y, filename):
    if y.dtype != 'int16':
        y = encode_16bits(y)
    scipy.io.wavfile.write(filename, sample_rate, y.astype(np.int16))


def load_mfcc(filename, encode=True):
    y, sr = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y=y, sr=8000, hop_length=64, n_mfcc=13)
    if encode:
        mfcc = encode_16bits(mfcc)
    return mfcc


def split_signal(x):
    unsigned = x + 2 ** 15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine


def combine_signal(coarse, fine):
    return coarse * 256 + fine - 2 ** 15


def encode_16bits(x):
    return np.clip(x * 2 ** 15, -2 ** 15, 2 ** 15 - 1).astype(np.int16)


mel_basis = None


def linear_to_mel(spectrogram):
    global mel_basis
    if mel_basis is None:
        mel_basis = build_mel_basis()
    return np.dot(mel_basis, spectrogram)


def build_mel_basis():
    return librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels, fmin=fmin)


def normalize(s):
    return np.clip((s - min_level_db) / -min_level_db, 0, 1)


def denormalize(s):
    return (np.clip(s, 0, 1) * -min_level_db) + min_level_db


def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    return np.power(10.0, x * 0.05)


def spectrogram(y):
    D = stft(y)
    S = amp_to_db(np.abs(D)) - ref_level_db
    return normalize(S)


def melspectrogram(y):
    D = stft(y)
    S = amp_to_db(linear_to_mel(np.abs(D)))
    return normalize(S)


def stft(y):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
