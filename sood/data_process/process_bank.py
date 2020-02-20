#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#


from __future__ import absolute_import, division, print_function, unicode_literals

from collections import defaultdict

from scipy.io import arff
import numpy as np
import json
from sood.data_process.data_loader import Dataset
from sood.util import Consts

# ==================================================
# DATASET bank
# Instance: 41188
# Feature: 53
# Class 1: 4640
# Class 0: 36548
# Source:
# ==================================================


arff_data = arff.loadarff("../dataset/bank/bank-additional-ful-nominal.arff")
data = []
dataset = Dataset(Dataset.BANK)
with open(dataset.file_path, "w") as w:
    attributes = defaultdict(lambda: len(attributes))
    for row in arff_data[0]:
        for idx, i in enumerate(row.tolist()[:-1]):
            attributes[f"{idx}_{i}"] += 0

    for row in arff_data[0]:
        row = row.tolist()

        record = [0] * len(attributes)
        assert f"{idx}_{i}" in attributes, f"{idx}_{i}"
        for idx, i in enumerate(row[:-1]):
            record[attributes[f"{idx}_{i}"]] = 1

        data.append(
            {
                Consts.DATA: record,
                Consts.LABEL: 1 if row[-1] == b"yes" else 0
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
print(f"# DATASET {Dataset.BANK}")
print(f"# Instance: {len(data)}")
print(f"# Feature: {len(data[0][Consts.DATA])}")
print(f"# Class 1: {class_1}")
print(f"# Class 0: {class_0}")
print(f"# Source: ")
print("# " + "=" * 50)
