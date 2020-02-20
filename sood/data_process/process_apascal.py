#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#


from __future__ import absolute_import, division, print_function, unicode_literals
from scipy.io import arff
import numpy as np
import json
from sood.data_process.data_loader import Dataset
from sood.util import Consts

arff_data = arff.loadarff("../dataset/apascal/apascal_entire_trainvsall.arff")
data = []
dataset = Dataset(Dataset.APASCAL)
with open(dataset.file_path, "w") as w:
    for row in arff_data[0]:
        row = row.tolist()
        _row = list(row[:-1])

        data.append(
            {
                Consts.DATA: [float(i) for i in _row],
                Consts.LABEL: 1 if row[-1] == 1 else 0
            }
        )

with open(dataset.file_path, "w") as w:
    for d in data:
        w.write(f"{json.dumps(d)}\n")

class_1 = 0
class_0 = 0
for i in data:
    if i[Consts.LABEL] == 1:
        class_1 += 1
    else:
        class_0 += 1

print("# " + "=" * 50)
print(f"# DATASET {Dataset.APASCAL}")
print(f"# Instance: {len(data)}")
print(f"# Feature: {len(data[0][Consts.DATA])}")
print(f"# Class 1: {class_1}")
print(f"# Class 0: {class_0}")
print(f"# Source: ")
print("# " + "=" * 50)

