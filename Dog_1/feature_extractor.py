import os
import re
import parser
import yasa
import h5py
import scipy.io as sio
import numpy as np
from tqdm import tqdm

# Point this to root directory with all patient folders.
data_dir = '/Volumes/My Passport/AI_Research/data/'


def get_bands():
    bands = [(0.1, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
             (12, 30, 'Beta'), (30, 70, 'Low Gamma'), (70, 180, 'High Gamma')]
    return bands


patient_names = [name for name in next(os.walk(data_dir))[1] if name != '.ipynb_checkpoints']
for i in range(len(patient_names)):
    patient_name = patient_names[i]
    files_dir = data_dir + patient_name + '/' + patient_name + '/'
    patient_files = os.listdir(files_dir)
    for j in tqdm(range(len(patient_files)), desc=patient_name):
        patient_file_name = patient_files[j]
        if not re.findall('_test_segment_', patient_file_name):
            matpath = files_dir + patient_file_name
            mat = sio.loadmat(matpath)
            data = parser.get_data(mat)
            channels = parser.get_channels(mat)
            channels = [channel.item() for channel in channels]
            fs = parser.get_sampling_frequency(mat)
            n = np.rint(fs).astype(int) * (60 * 10) # select all 10 mins in the mat file 
            t = np.arange(0, n) / fs
            X = data[0:n]
            bands = get_bands()
            bp = yasa.bandpower(X, fs, win_sec=20,
                           bands=bands, bandpass=True, ch_names=channels, relative=False).drop(labels=['TotalAbsPow', 'FreqRes', 'Relative'], axis=1).to_numpy()
            save_dir = data_dir + patient_name + '/' + 'Power_In_Band_Features/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_name = save_dir + patient_file_name.split('.')[0] + '_pib' + '.h5'
            f = h5py.File(save_name, 'w')
            data_header = patient_file_name.split('.')[0] + '_pib'
            f.create_dataset(data_header, data=bp)
            f.close()