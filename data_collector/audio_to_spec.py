import numpy as np
import librosa
import os
import warnings
import re

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)


audio_conf = {}
audio_type = audio_conf.get('audio_type', 'melspectrogram')
preemph_coef = audio_conf.get('preemph_coef', 0.97)
sample_rate = audio_conf.get('sample_rate', 16000)
window_size = audio_conf.get('window_size', 0.025)
window_stride = audio_conf.get('window_stride', 0.01)
window_type = audio_conf.get('window_type', 'hamming')
num_mel_bins = audio_conf.get('num_mel_bins', 40)
target_length = audio_conf.get('target_length', 300)
use_raw_length = audio_conf.get('use_raw_length', False)
padval = audio_conf.get('padval', 0)
fmin = audio_conf.get('fmin', 20)
ref = audio_conf.get('red', 20)
max_db = audio_conf.get('max_db', 100)
n_fft = audio_conf.get('n_fft', int(sample_rate * window_size))
win_length = int(sample_rate * window_size)
hop_length = int(sample_rate * window_stride)

warnings.filterwarnings("ignore", message='PySoundFile failed. Trying audioread instead.')

audio_dir = '../data/audio_files/'

audio_files = sorted_alphanumeric(os.listdir(audio_dir))
print(audio_files)

specs = []
for idx, path in enumerate(audio_files):
    y, sr = librosa.load(audio_dir + path, sample_rate)
    #if y.size == 0:
    #    y = np.zeros(200)
    #y = y - y.mean()
    #y = preemphasis(y, preemph_coef)

    #-----------------
    spec = librosa.feature.melspectrogram(y=y, sr=sr)
    spec = librosa.power_to_db(spec, ref = 20)
    logspec = np.clip((spec + max_db) / max_db, 0, 1)
    n_frames = logspec.shape[1]
    if use_raw_length:
        target_length = n_frames
    p = target_length - n_frames
    if p > 0:
        logspec = np.pad(logspec, ((0,0),(0,p)), 'constant',
            constant_values=(padval,padval))
    elif p < 0:
        logspec = logspec[:,0:p]
        n_frames = target_length
    specs.append(logspec)

specs = np.asarray(specs)
np.save('../data/specs.npy', specs)
