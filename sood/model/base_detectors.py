#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from sklearn.neighbors import LocalOutlierFactor



def normalize_z_score(score: np.array):
    # =========================================================
    # Positive large z-score indicates anomaly
    # =========================================================
    mean = np.mean(score)
    std = np.std(score)
    return (score - mean) / std

class LOF:
    def __init__(self, neighbor, if_normalize):
        self.neighor = neighbor
        self.if_normalize = if_normalize
        self.lof = LocalOutlierFactor(n_neighbors=10, metric="euclidean")

    def fit(self, data: np.array):
        # =========================================================
        # Positive large indicates anomaly
        # =========================================================
        self.lof.fit(data)
        if self.if_normalize:
            return normalize_z_score(-self.lof.negative_outlier_factor_)
        else:
            return -self.lof.negative_outlier_factor_
