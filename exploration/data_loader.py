import os
import re
import numpy as np

def load_subject(subject_name="Dog_1"):
    root = './data/'
    root_children = os.listdir(root)
    subject_idx = root_children.index(subject_name)
    subject_path = os.path.join(root, root_children[subject_idx])
    subject_path_2 = os.path.join(subject_path, os.listdir(subject_path)[0])
    return os.listdir(subject_path_2), subject_path_2

def select_segment(subject_files=None, segment_name='interictal'):
    segment_files = []
    for subject_file in subject_files:
        if re.findall(segment_name, subject_file):
            segment_files.append(subject_file)
    return segment_files
