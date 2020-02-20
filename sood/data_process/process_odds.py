#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
from scipy.io import loadmat
from sood.data_process.data_loader import Dataset
import numpy as np
from sood.util import PathManager, Consts
from sood.log import getLogger
import json

# ==================
# Dataset: Arrhythmia
# Instance: 452
# Feature: 274
# Class 1: 66
# Class 0: 386
# Source: http://odds.cs.stonybrook.edu/arrhythmia-dataset/
# ==================

# ==================
# Dataset: MNIST
# Instance: 7603
# Feature: 100
# Class 1: 700
# Class 0: 6903
# Source: http://odds.cs.stonybrook.edu/mnist-dataset/
# ==================

# ==================
# Dataset: MUSK
# Instance: 3062
# Feature: 166
# Class 1: 97
# Class 0: 2965
# Source: http://odds.cs.stonybrook.edu/musk-dataset/
# ==================

# ==================
# Dataset: OPTDigits
# Instance: 5216
# Feature: 64
# Class 1: 150
# Class 0: 5066
# Source: http://odds.cs.stonybrook.edu/optdigits-dataset/
# ==================

# ==================
# Dataset: PIMA
# Instance: 768
# Feature: 8
# Class 1: 268
# Class 0: 500
# Source: http://odds.cs.stonybrook.edu/pima-indians-diabetes-dataset/
# ==================

# ==================
# Dataset: SPEECH
# Instance: 768
# Instance: 3686
# Feature: 400
# Class 1: 61
# Class 0: 3625
# Source: http://odds.cs.stonybrook.edu/speech-dataset/
# ==================

# ==================
# Dataset: Thyroid
# Instance: 3772
# Feature: 6
# Class 1: 93
# Class 0: 3679
# Source: http://odds.cs.stonybrook.edu/thyroid-disease-dataset/
# ==================

# ==================
# Dataset: Glass
# Instance: 214
# Feature: 9
# Class 1: 9
# Class 0: 205
# Source: http://odds.cs.stonybrook.edu/glass-data/
# ==================

# ==================================================
# DATASET Shuttle
# Instance: 49097
# Feature: 9
# Class 1: 3511
# Class 0: 45586.0
# Source: http://odds.cs.stonybrook.edu/shuttle-dataset/
# ==================================================

# ==================================================
# DATASET Breastw
# Instance: 683
# Feature: 9
# Class 1: 239
# Class 0: 444.0
# Source:
# ==================================================

# ==================================================
# DATASET Ecoli
# Instance: 336
# Feature: 7
# Class 1: 0
# Class 0: 336.0
# Source:
# ==================================================

# ==================================================
# DATASET Mammography
# Instance: 11183
# Feature: 11183
# Class 1: 260
# Class 0: 10923.0
# Source:
# ==================================================

# ==================================================
# DATASET Annthyroid
# Instance: 7200
# Feature: 7200
# Class 1: 6
# Class 0: 6666.0
# Source: 
# ==================================================

# ==================================================
# DATASET Vertebral
# Instance: 240
# Feature: 6
# Class 1: 30
# Class 0: 210.0
# Source:
# ==================================================


logger = getLogger(__name__)

path_manager = PathManager()
dataset_name = Dataset.VERTEBRAL
dataset = Dataset(dataset_name)

speech = loadmat(dataset.mat_file_path)
print("# " + "=" * 50)
print(f"# DATASET {dataset_name}")
print(f"# Instance: {speech['X'].shape[0]}")
print(f"# Feature: {speech['X'].shape[1]}")
print(f"# Class 1: {np.sum(speech['y'])}")
print(f"# Class 0: {speech['y'].shape[0] - np.sum(speech['y'])}")
print(f"# Source: ")
print("# " + "=" * 50)

logger.info(speech.keys())
data = []
for i in speech['X']:
    data.append(
        {Consts.DATA: i.tolist()}
    )

for idx, i in enumerate(speech['y']):
    data[idx][Consts.LABEL] = int(i)

with open(dataset.file_path, "w") as w:
    for d in data:
        w.write(f"{json.dumps(d)}\n")

logger.info(f"Output Path: {dataset.file_path}")
