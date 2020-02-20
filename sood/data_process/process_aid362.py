#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Instances: 2584
# Attributes: 18 + class
# Class distribution:
#     hazardous state" (class 1)    :  170  (6.6%)
#     "non-hazardous state" (class 0): 2414 (93.4%)


from __future__ import absolute_import, division, print_function, unicode_literals
from scipy.io import arff
import numpy as np
import json
from sood.data_process.data_loader import Dataset
from sood.util import Consts

arff_data = arff.loadarff("../dataset/aid362/AID362red_train_allpossiblenominal.arff")
data = []
dataset = Dataset(Dataset.AID362)
with open(dataset.file_path, "w") as w:
    for row in arff_data[0]:
        row = row.tolist()
        _row = list(row[:-2])
        last_attr = [0, 0, 0, 0]
        last_attr[int(row[-2])] = 1
        _row = _row + last_attr

        data.append(
            {
                Consts.DATA: [float(i) for i in _row],
                Consts.LABEL: 1 if row[-1] == b"Active" else 0
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
print(f"# DATASET {Dataset.AID362}")
print(f"# Instance: {len(data)}")
print(f"# Feature: {len(data[0][Consts.DATA])}")
print(f"# Class 1: {class_1}")
print(f"# Class 0: {class_0}")
print(f"# Source: ")
print("# " + "=" * 50)

