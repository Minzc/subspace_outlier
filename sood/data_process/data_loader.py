#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import json
from sood.log import getLogger

from sood.util import PathManager, Consts

logger = getLogger(__name__)


class Dataset:
    ARRHYTHMIA = "arrhythmia"

    def __init__(self, dataset):
        path_manager = PathManager()
        self.dataset_name = dataset
        self.file_path = f"{path_manager.dataset}/{dataset}/{dataset}.json"


class DataLoader:
    @staticmethod
    def load(dataset_name: str):
        if dataset_name == Dataset.ARRHYTHMIA:
            dataset = Dataset(dataset_name)
        else:
            raise Exception(f"The dataset {dataset_name} is not support")
        X = []
        Y = []
        with open(dataset.file_path) as f:
            for ln in f:
                obj = json.loads(ln.strip())
                X.append(obj[Consts.DATA])
                Y.append(obj[Consts.LABEL])
        logger.debug(f"Dataset name: {dataset_name}")
        logger.debug(f"Dataset dimension: {np.array(X).shape}")
        logger.debug(f"Label dimension: {np.array(Y).shape}")
        return np.array(X), np.array(Y)


if __name__ == '__main__':
    DataLoader.load(Dataset.ARRHYTHMIA)
