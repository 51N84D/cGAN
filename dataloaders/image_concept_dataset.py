import json
import librosa
import numpy as np
import os
from PIL import Image
import scipy.signal
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import warnings


#Adapted from https://github.com/dharwath/DAVEnet-pytorch
def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """    
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class ImageConceptDataset(Dataset):
    def __init__(self, dataset_paths):
        """
        Dataset that manages audio and 
        """
        self.images=np.load(dataset_paths['image'])
        self.labels=np.load(dataset_paths['labels'])
        self.audio=np.load(dataset_paths['audio'])
    
        self.audio_params = {}
        self.image_params = {}

        RGB_mean = self.image_params.get('RGB_mean', [0.485, 0.456, 0.406])
        RGB_std = self.image_params.get('RGB_std', [0.229, 0.224, 0.225])
        self.image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

    #Use if dataset becomes too large to preprocess upfront
    def loadAudio(self, path):
        audio_type = self.audio_params.get('audio_type', 'melspectrogram')
        preemph_coef = self.audio_params.get('preemph_coef', 0.97)
        sample_rate = self.audio_params.get('sample_rate', 16000)
        window_size = self.audio_params.get('window_size', 0.025)
        window_stride = self.audio_params.get('window_stride', 0.01)
        window_type = self.audio_params.get('window_type', 'hamming')
        num_mel_bins = self.audio_params.get('num_mel_bins', 40)
        target_length = self.audio_params.get('target_length', 300)
        use_raw_length = self.audio_params.get('use_raw_length', False)
        padval = self.audio_params.get('padval', 0)
        fmin = self.audio_params.get('fmin', 20)
        ref = self.audio_params.get('red', 20)
        max_db = self.audio_params.get('max_db', 100)
        n_fft = self.audio_params.get('n_fft', int(sample_rate * window_size))
        win_length = int(sample_rate * window_size)
        hop_length = int(sample_rate * window_stride)

        # load audio, subtract DC, preemphasis
        warnings.filterwarnings("ignore", message='PySoundFile failed. Trying audioread instead.')
        y, sr = librosa.load(path, sample_rate)

        #-----------------
        spec = librosa.feature.melspectrogram(y=y, sr=sr)
        spec = librosa.power_to_db(spec, ref = 20)
        logspec = np.clip((spec + max_db) / max_db, 0, 1)
        #------------------
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
        logspec = torch.FloatTensor(logspec)
        return logspec
    
    def loadImage(self, img):
        img = np.transpose(img, (2,0,1))
        img = torch.FloatTensor(img)
        #normalize to -1, 1
        img = (2*img / 255.0) - 1.0
        return img
    

    
    def __getitem__(self, index):
        """
        returns: image, label, audio
        -image is a FloatTensor of size (3, H, W)
        -audio is a FloatTensor spectrogram of size (N_freq, N_frames)
        """
        im = np.squeeze(self.images[index])
        image = self.loadImage(im)
        label = np.squeeze(self.labels[index])
        #audio = self.loadAudio(self.audio_path + str(label) + '.wav')
        audio = self.audio[label]
        return (image, label, audio)
    
    def __len__(self):
        return self.images.shape[0]





