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
# Instance: 3772
# Feature: 6
# Class 1: 93
# Class 0: 3679
# Source: http://odds.cs.stonybrook.edu/thyroid-disease-dataset/
# ==================

logger = getLogger(__name__)

path_manager = PathManager()
dataset = Dataset(Dataset.THYROID)

speech = loadmat(dataset.mat_file_path)
logger.info(speech.keys())
logger.info(speech['X'].shape)
logger.info(speech['y'].shape)
logger.info(f"Anomaly size: {np.sum(speech['y'])} Inlier size: {speech['y'].shape[0] - np.sum(speech['y'])}")

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