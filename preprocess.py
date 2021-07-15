import os
import re
import numpy as np
from scipy.io import loadmat
from scipy import signal
from tqdm import tqdm

def is_interictal(name):
    return 'interictal' in name

def is_preictal(name):
    return 'preictal' in name

def get_preictal_files(directory_path):
    preictal_filter = filter(is_preictal, os.listdir(dir_path))
    preictals = [preictal for preictal in preictal_filter]
    return preictals

def get_interictal_files(directory_path):
    interictal_filter = filter(is_interictal, os.listdir(dir_path))
    interictals = [interictal for interictal in interictal_filter]
    return interictals

def get_sig(mat):
    "Returns (channel x times) matrix representing EEG signal in microvolt."
    segment_name = list(mat.keys())[-1]
    segment = mat[segment_name]
    sig = segment['data'][0][0]
    return sig

def get_fs(mat):
    "Returns the sampling frequency."
    segment_name = list(mat.keys())[-1]
    segment = mat[segment_name]
    fs = segment['sampling_frequency'][0][0][0][0]
    return fs

def stft(mat, window_size):
    "Performs Short Time Fourier Transform on EEG signal with Hann window of specified length."
    sigbuf = get_sig(mat)
    fs = np.round(get_fs(mat))
    n_channels = sigbuf.shape[0]
    print('Raw EEG matrix shape:', sigbuf.shape)
    window_idx = np.rint(window_size/(1/fs)).astype(int)
    freq_len = (window_idx/2) + 1
    time_len = np.max(t).astype(int)
    Zxxs = np.zeros((n_channels, freq_len, time_len), dtype='complex')
    print('STFT matrix shape', Zxxs.shape)
    for i in tqdm(range(1, sigbuf.shape[0])):
        sig = sigbuf[i,:]
        timewin = 2 # sliding window duration in seconds
        timewinidx = np.rint(timewin/(1/fs)).astype(int)
        _, _, Zxx = signal.stft(sig, fs=fs, window='hann', nperseg=window_idx, noverlap=None, return_onesided=True)
        Zxxs[i,:] = Zxx
    return Zxxs

