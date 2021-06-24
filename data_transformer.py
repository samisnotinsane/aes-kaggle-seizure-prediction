import re
import data_loader as loader
from tqdm.auto import trange
import numpy as np

def keep_interictal(subject_files):
  segment_files = []
  for i in trange(len(subject_files), desc="keep_interictal"):
    subject_file = subject_files[i]
    if re.findall('interictal', subject_file):
      segment_files.append(subject_file)
  return segment_files

def keep_preictal(subject_files):
  segment_files = []
  for i in trange(len(subject_files), desc="keep_preictal"):
    subject_file = subject_files[i]
    if re.findall('preictal', subject_file):
      segment_files.append(subject_file)
  return segment_files

def exclude_test(subject_files):
  interictals = keep_interictal(subject_files)
  preictals = keep_preictal(subject_files)
  return interictals + preictals

def chrono_segment(segments):
  sequence_nos = [segment.split('.')[0].split('_')[-1] for segment in segments]
  sorted_sequence_segment = sorted(list(zip(sequence_nos, segments)))
  return [segment[1] for segment in sorted_sequence_segment]

def label_segment(subject_files, label):
  targets = None
  for i in range(len(subject_files)):
    if label == 0:
      targets = np.zeros((len(subject_files)))
    if label == 1:
      targets = np.ones((len(subject_files)))
  return targets

def make_input_matrix(subject_files, path):
    input_files = exclude_test(subject_files)
    X = np.array([path + '/' + file for file in input_files])
    return X

def make_input_target():
    subject_names = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    subject_paths = []
    arr_inputs = []
    arr_targets = []
    for i in trange(len(subject_names), desc='Subject'):
        files, path = loader.load_subject(subject_names[i])
        subject_paths.append(path)
        interictal_inputs = keep_interictal(files)
        arr_inputs.append([path + '/' + interictal_input for interictal_input in interictal_inputs])
        
        preictal_inputs = keep_preictal(files)
        arr_inputs.append([path + '/' + preictal_input for preictal_input in preictal_inputs])
        
        arr_targets.append(label_segment(interictal_inputs, 0))
        arr_targets.append(label_segment(preictal_inputs, 1))        
    X = np.concatenate(arr_inputs).ravel()
    y = np.concatenate(arr_targets).ravel()
    y = y.astype(int)
    return X, y
