#!/usr/local/bin/python

import os

patient_dir = 'Dog_1/Dog_1/'
files = os.listdir(patient_dir)

preictal_count = 0
interictal_count = 0

for filename in files:
  if os.path.isfile(os.path.join(patient_dir, filename)):
    name = filename.split('.')[0]
    segment_type = name.split('_')[2]
    if segment_type == 'preictal':
      preictal_count += 1
    if segment_type == 'interictal':
      interictal_count += 1
print('Preictal count:', preictal_count)
print('Interictal count:', interictal_count)

