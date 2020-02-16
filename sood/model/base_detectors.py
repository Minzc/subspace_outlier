#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestNeighbors
from sood.data_process.data_loader import DataLoader, Dataset

from sood.log import getLogger

logger = getLogger(__name__)


def normalize_z_score(score: np.array):
    # =========================================================
    # Positive large z-score indicates anomaly
    # =========================================================
    mean = np.mean(score)
    std = np.std(score)
    if std != 0:
        return (score - mean) / std
    else:
        return score - mean

class LOF:
    NAME = "LOF"
    def __init__(self, neighbor, if_normalize):
        self.neighor = neighbor
        self.if_normalize = if_normalize
        self.lof = LocalOutlierFactor(n_neighbors=neighbor, metric="euclidean")

    def fit(self, data: np.array):
        # =========================================================
        # Positive large indicates anomaly
        # =========================================================
        self.lof.fit(data)
        if self.if_normalize:
            return normalize_z_score(-self.lof.negative_outlier_factor_)
        else:
            return -self.lof.negative_outlier_factor_

class kNN:
    NAME = "kNN"
    def __init__(self, neighbor, if_normalize):
        self.neighbor = neighbor
        self.if_normalize = if_normalize

    def fit(self, data: np.array):
        clf = NearestNeighbors(self.neighbor)
        clf.fit(data)
        score, indices = clf.kneighbors(data)

        score = np.mean(score, axis=1)
        logger.debug(f"KNN score shape : {score.shape}")
        if self.if_normalize:
            return normalize_z_score(score)
        else:
            return score

def test():
    import json
    from sood.util import PathManager, Consts
    import scipy.io as sio
    dataset = Dataset(Dataset.MUSK)
    data = sio.loadmat("data/musk.mat")
    X_train = data['X'].astype('double')
    y_label = np.squeeze(data['y']).astype('int')
    neigh = max(10, int(np.floor(0.03 * X_train.shape[0])))

    knn = kNN(neigh, if_normalize=False)
    rst = knn.fit(X_train)
    y_scores = np.array(rst)
    print(y_scores)
    roc_auc = roc_auc_score(y_label, y_scores)
    print(roc_auc)

