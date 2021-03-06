#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import json

from pyod.utils import standardizer

from sood.log import getLogger

from sood.util import PathManager, Consts

logger = getLogger(__name__)


class Dataset:
    ARRHYTHMIA = "arrhythmia"
    SPEECH = "speech"
    MUSK = "musk"
    MNIST_ODDS = "mnist_odds"
    OPTDIGITS = "optdigits"
    THYROID = "thyroid"
    PIMA = "pima"
    GLASS = "glass"
    SHUTTLE = "shuttle"
    BREASTW = "breastw"
    ECOLI = "ecoli"
    MAMMOGRAPHY = "mammography"
    ANNTHYROID = "annthyroid"
    VERTEBRAL = "vertebral"
    SEISMIC = "seismic"
    WINE = "wine"
    VOWELS = "vowels"
    AD = "ad"
    APASCAL = "apascal"
    AID362 = "aid362"
    BANK = "bank"
    PROB = "prob"
    U2R = "u2r"

    def __init__(self, dataset):
        path_manager = PathManager()
        self.dataset_name = dataset
        self.file_path = f"{path_manager.dataset}/{dataset}/{dataset}.json"
        self.mat_file_path = f"{path_manager.dataset}/{dataset}/{dataset}.mat"

    @classmethod
    def supported_dataset(cls):
        return [cls.ANNTHYROID, cls.VERTEBRAL, cls.APASCAL, cls.AID362,
                cls.MAMMOGRAPHY, cls.ECOLI, cls.BREASTW, cls.SHUTTLE,
                cls.MNIST_ODDS, cls.VERTEBRAL, cls.SEISMIC, cls.WINE,
                cls.ARRHYTHMIA, cls.SPEECH, cls.MUSK, cls.OPTDIGITS,
                cls.THYROID, cls.PIMA, cls.GLASS, cls.VOWELS, cls.AD, cls.AID362, cls.BANK, cls.PROB, cls.U2R]


class DataLoader:
    @staticmethod
    def load(dataset_name: str):
        if dataset_name in Dataset.supported_dataset():
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
        X = np.array(X)
        return X, np.array(Y)


if __name__ == '__main__':
    DataLoader.load(Dataset.ARRHYTHMIA)
