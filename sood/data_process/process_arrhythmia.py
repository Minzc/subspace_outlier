#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
from scipy.io import loadmat
from sood.util import PathManager, Consts
import json
from collections import defaultdict

# ==================
# Instance: 452
# Feature: 274
# Class 1: 66
# Class 0: 386
# ==================

path_manager = PathManager()

arrhythmia = loadmat(f'{path_manager.dataset}/arrhythmia/arrhythmia.mat')
print(arrhythmia.keys())
print(arrhythmia['X'].shape)
print(arrhythmia['y'].shape)

data = []
for i in arrhythmia['X']:
    data.append(
        {Consts.DATA: i.tolist()}
    )

for idx, i in enumerate(arrhythmia['y']):
    data[idx][Consts.LABEL] = int(i)

output_file = f'{path_manager.dataset}/arrhythmia/arrhythmia.json'
with open(output_file, "w") as w:
    for d in data:
        w.write(f"{json.dumps(d)}\n")

print(f"Output Path: {output_file}")