#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
from scipy.io import loadmat
from sood.data_process.data_loader import Dataset

from sood.util import PathManager, Consts
from sood.log import getLogger
import json
from collections import defaultdict

# ==================
# Instance: 452
# Feature: 274
# Class 1: 66
# Class 0: 386
# ==================

logger = getLogger(__name__)

path_manager = PathManager()
dataset = Dataset(Dataset.ARRHYTHMIA)

arrhythmia = loadmat(dataset.mat_file_path)
logger.info(arrhythmia.keys())
logger.info(arrhythmia['X'].shape)
logger.info(arrhythmia['y'].shape)

data = []
for i in arrhythmia['X']:
    data.append(
        {Consts.DATA: i.tolist()}
    )

for idx, i in enumerate(arrhythmia['y']):
    data[idx][Consts.LABEL] = int(i)


with open(dataset.file_path, "w") as w:
    for d in data:
        w.write(f"{json.dumps(d)}\n")

logger.info(f"Output Path: {dataset.file_path}")