#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

from sood.data_process.data_loader import Dataset
import json
from sood.log import getLogger
from sood.util import PathManager, Consts

# ==================================================
# DATASET ad
# Instance: 3279
# Feature: 1555
# Class 1: 459
# Class 0: 2820
# Source:
# ==================================================


logger = getLogger(__name__)

path_manager = PathManager()
dataset_name = Dataset.AD
dataset = Dataset(dataset_name)

data = []
class_1 = 0
class_0 = 0
with open("../dataset/ad/ad.data") as f:
    for ln in f:
        ln_segs = ln.strip().split(",")
        ln_segs = ln_segs[3:]
        assert len(ln_segs) == 1556
        data.append(
            {
                Consts.DATA: [int(i) if i != "?" else 0 for i in ln_segs[:-1]],
                Consts.LABEL: 1 if ln_segs[-1] == "ad." else 0
            }
        )
        if data[-1][Consts.LABEL] == 1:
            class_1 +=1
        else:
            class_0 += 1



print("# " + "=" * 50)
print(f"# DATASET {dataset_name}")
print(f"# Instance: {len(data)}")
print(f"# Feature: {len(data[0][Consts.DATA])}")
print(f"# Class 1: {class_1}")
print(f"# Class 0: {class_0}")
print(f"# Source: ")
print("# " + "=" * 50)


with open(dataset.file_path, "w") as w:
    for d in data:
        w.write(f"{json.dumps(d)}\n")

logger.info(f"Output Path: {dataset.file_path}")