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

arff_data = arff.loadarff("../dataset/seismic/seismic-bumps.arff")
data = []
dataset = Dataset(Dataset.SEISMIC)
with open(dataset.file_path, "w") as w:
    for row in arff_data[0]:
        data.append(
            {
                Consts.DATA_SET: [i for i in row if np.isreal(i)]
            }
        )
    for idx, row in enumerate(arff_data[0]):
        data[idx][Consts.LABEL] = int(row[-1])

with open(dataset.file_path, "w") as w:
    for d in data:
        w.write(f"{json.dumps(d)}\n")

