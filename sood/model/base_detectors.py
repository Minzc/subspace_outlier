#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted, check_array

from sood.data_process.data_loader import DataLoader, Dataset
from pyod.models.knn import KNN
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


class kNN(KNN):
    NAME = "kNN"

    def __init__(self, neighbor, if_normalize, selected_features: np.array = None):
        super().__init__(n_neighbors=neighbor, method='mean', metric='euclidean')
        self.neighbor = neighbor
        self.if_normalize = if_normalize
        self.selected_features = selected_features

    def fit(self, data, y=None):
        if self.selected_features is not None:
            logger.info(f"Before data shape {data.shape}")
            data = data[:, self.selected_features]
            logger.info(f"After data shape {data.shape}")

        super().fit(data)
        score = self.decision_scores_

        logger.debug(f"KNN score shape : {score.shape}")
        if self.if_normalize:
            return normalize_z_score(score)
        else:
            return score

    def decision_function(self, X):
        logger.info(f"Before data shape {X.shape}")
        X = X[:, self.selected_features]
        logger.info(f"After data shape {X.shape}")
        return super().decision_function(X)

class GKE:
    NAME = "GKE"

    def __init__(self, neighbor, if_normalize):
        self.if_normalize = if_normalize

    def fit(self, data: np.array):
        stds = np.std(data, axis=0)
        for idx, i in enumerate(stds):
            if i == 0:
                data[:, idx] = 0
                stds[idx] = 1
        data = data / stds
        def compute(u, v):
            return ((u - v) ** 2).sum()
        pairwise_distance = cdist(data, data, compute)
        pairwise_distance = np.exp(-pairwise_distance)
        pairwise_distance = np.sum(pairwise_distance, axis=1)
        score = pairwise_distance
        if self.if_normalize:
            return normalize_z_score(-score)
        else:
            return -score

def test():
    X, Y = DataLoader.load(Dataset.MUSK)
    neigh = max(10, int(np.floor(0.03 * X.shape[0])))
    knn = kNN(neigh, if_normalize=False)
    rst = knn.fit(X)
    y_scores = np.array(rst)
    print(y_scores)
    roc_auc = roc_auc_score(Y, y_scores)
    print(roc_auc)


if __name__ == '__main__':
    test()
