import data_loader as data
import data_transformer as transformer
from scipy.io import loadmat
from tqdm.auto import trange

def check_channels(subject_names):
  subject_names = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
  subject_item_count = []
  for i in trange(len(subject_names), desc='Subject'):
    subject_name = subject_names[i]
    files, path = data.load_subject(subject_name)
    interictal = data.select_segment(files, 'interictal')
    preictal = data.select_segment(files, 'preictal')
    ffiles = interictal + preictal
    item_count = []
    for j in trange(len(ffiles), desc='Segment'):
        fpath = path + '/' + ffiles[j]
        mat = loadmat(fpath)
        segment_label = list(mat.keys())[-1]
        X = mat[segment_label]
        item_count.append(X['channels'][0][0][0].shape[0])
    subject_item_count.append(set(item_count))
  return subject_item_count, subject_names

def check_frequency(subject_files, path):
    ffiles = transformer.exclude_test(subject_files)
    item_count = []
    for i in trange(len(ffiles), desc='check_frequency'):
        fpath = path + '/' + ffiles[i]
        mat = loadmat(fpath)
        segment_label = list(mat.keys())[-1]
        X = mat[segment_label]
        item_count.append(X['sampling_frequency'][0][0][0][0])
    return set(item_count)
