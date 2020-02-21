#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sood.data_process.data_loader import DataLoader, Dataset
from pyod.models.knn import KNN
from sood.log import getLogger
from sood.util import Normalize

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

    def __init__(self, neighbor_size: int, norm_method: str):
        assert norm_method in {Normalize.UNIFY, Normalize.ZSCORE, None}
        self.neighbor_size = neighbor_size
        self.norm_method = norm_method
        self.lof = LocalOutlierFactor(n_neighbors=neighbor_size, metric="euclidean")

    def fit(self, data: np.array):
        # =========================================================
        # Positive large indicates anomaly
        # =========================================================
        self.lof.fit(data)
        if self.norm_method == Normalize.ZSCORE:
            return Normalize.zscore(-self.lof.negative_outlier_factor_)
        elif self.norm_method == Normalize.UNIFY:
            return Normalize.unify(-self.lof.negative_outlier_factor_)
        else:
            return -self.lof.negative_outlier_factor_


class kNN_GPU:
    NAME = "kNN"

    def __init__(self, neighbor_size: int, norm_method: str):
        assert norm_method in {Normalize.UNIFY, Normalize.ZSCORE, None}
        self.neighbor_size = neighbor_size
        self.norm_method = norm_method

    def fit(self, data, y=None):
        import torch
        assert torch.cuda.is_available()
        data = torch.tensor(data, device="cuda:0", dtype=torch.float64)
        dist_matrix = torch.norm(data[:, None] - data, dim=2, p=2)
        top_k_nearest = torch.topk(dist_matrix, k=self.neighbor_size + 1, dim=1, largest=False)[0]
        score = torch.sum(top_k_nearest, dim=1).numpy()

        logger.debug(f"KNN score matrix: {dist_matrix.shape}")

        if self.norm_method == Normalize.ZSCORE:
            return Normalize.zscore(score)
        elif self.norm_method == Normalize.UNIFY:
            return Normalize.unify(score)
        else:
            return score

class kNN(KNN):
    NAME = "kNN"

    def __init__(self, neighbor_size: int, norm_method: str):
        super().__init__(n_neighbors=neighbor_size, method='mean', metric='euclidean')
        assert norm_method in {Normalize.UNIFY, Normalize.ZSCORE, None}
        self.neighbor_size = neighbor_size
        self.norm_method = norm_method

    def fit(self, data, y=None):
        super().fit(data)
        score = self.decision_scores_

        logger.debug(f"KNN score shape : {score.shape}")
        if self.norm_method == Normalize.ZSCORE:
            return Normalize.zscore(score)
        elif self.norm_method == Normalize.UNIFY:
            return Normalize.unify(score)
        else:
            return score


class kNN_LSCP(KNN):
    NAME = "kNN_LSCP"

    def __init__(self, neighbor_size: int, selected_features: np.array):
        super().__init__(n_neighbors=neighbor_size, method='mean', metric='euclidean')
        self.neighbor_size = neighbor_size
        self.selected_features = selected_features

    def fit(self, data, y=None):
        logger.debug(f"Before data shape {data.shape}")
        data = data[:, self.selected_features]
        logger.debug(f"After data shape {data.shape}")
        return super().fit(data)


    def decision_function(self, X):
        logger.info(f"Before data shape {X.shape}")
        X = X[:, self.selected_features]
        logger.info(f"After data shape {X.shape}")
        return super().decision_function(X)

class GKE_GPU:
    NAME = "GKE"

    def __init__(self, norm_method: str):
        self.norm_method = norm_method

    @staticmethod
    def convert_to_tensor(data: np.array):
        stds = np.std(data, axis=0)
        stds[stds == 0] = 1
        n = np.math.pow(data.shape[0], 1.0/3)
        stds = stds * n * 1.06

        data = data / stds
        data = torch.tensor(data, dtype=torch.float64, device="cuda:0")
        return data

    def fit(self, data: torch.tensor):
        pairwise_distance = torch.cdist(data, data, 2)
        pairwise_distance = pairwise_distance * pairwise_distance * -1
        pairwise_distance = torch.exp(pairwise_distance)
        pairwise_distance = torch.sum(pairwise_distance, axis=1)
        score = -pairwise_distance
        score = score.cpu().numpy()

        if self.norm_method == Normalize.ZSCORE:
            return Normalize.zscore(score)
        elif self.norm_method == Normalize.UNIFY:
            return Normalize.unify(score)
        else:
            return score

# class GKE_EST:
#     NAME = "GKE"
#
#     def __init__(self, norm_method: str, neighbor):
#         self.norm_method = norm_method
#
#     def fit(self, data: np.array):
#         stds = np.std(data, axis=0)
#         for idx, i in enumerate(stds):
#             if i == 0:
#                 data[:, idx] = 0
#                 stds[idx] = 1
#         data = data / stds
#
#         pairwise_distance = torch.cdist(data, data, 2)
#         pairwise_distance = pairwise_distance * pairwise_distance * -1
#         pairwise_distance = torch.exp(pairwise_distance)
#         pairwise_distance = torch.sum(pairwise_distance, axis=1)
#         score = -pairwise_distance
#
#         if self.norm_method == Normalize.ZSCORE:
#             return Normalize.zscore(score)
#         elif self.norm_method == Normalize.UNIFY:
#             return Normalize.unify(score)
#         else:
#             return score

def test():
    X = np.array([
        [1,2],
        [3,4],
        [5,6]
    ])
    mdl = GKE_GPU(None)
    X = GKE_GPU.convert_to_tensor(X)
    selected_features = np.array([0,1])
    X = X[:, selected_features]
    score = mdl.fit(X)
    print(score)
    print(score.numpy())
    print(score.tolist())


if __name__ == '__main__':
    test()
