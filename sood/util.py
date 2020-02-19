#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

import os
from typing import List

from pyod.utils import wpearsonr
from scipy.stats import spearmanr
import numpy as np
from scipy.special._ufuncs import erf


class Consts:
    DATA = "data"
    LABEL = "label"
    ROC_AUC = "roc_aucs"
    PRECISION_A_N = "precision@n"
    TIME = "time"

    DATA_SET = "dataset"
    AGGREGATE = "aggregate"
    BASE_MODEL = "base_model"
    NEIGHBOR_SIZE = "neighbor_size"
    ENSEMBLE_SIZE = "ensemble_size"
    START_DIM = "start_dim"
    END_DIM = "end_dim"

class Similarity:
    @staticmethod
    def spearman(target_list: np.array, local_list: np.array) -> float:
        # Sort in descending order
        target_outlying_idx = np.argsort(target_list)[::-1]
        local_outlying_idx = np.argsort(local_list)[::-1]

        g_rank = [0] * target_outlying_idx.shape[0]
        l_rank = [0] * local_outlying_idx.shape[0]

        for rank, idx in enumerate(target_outlying_idx):
            g_rank[idx] = rank
        for rank, idx in enumerate(local_outlying_idx):
            l_rank[idx] = rank
        return spearmanr(g_rank, l_rank)[0]

    @staticmethod
    def pearson(target_list, local_list, if_weighted: bool) -> float:
        # Sort in descending order
        target_outlying_idx = np.argsort(target_list)[::-1]

        weights = [0] * target_outlying_idx.shape[0]
        for rank, idx in enumerate(target_outlying_idx):
            weights[idx] = 1 / (rank + 1)

        assert target_list[target_outlying_idx[0]] >= target_list[target_outlying_idx[1]],\
            f"Score {target_list[target_outlying_idx[0]]} {target_list[target_outlying_idx[1]]}"
        if if_weighted:
            score = wpearsonr(target_list, local_list, w=weights)
        else:
            score = wpearsonr(target_list, local_list, w=weights)
        if np.isnan(score) == True:
            return 0
        return score

class Normalize:
    ZSCORE = "zscore"
    UNIFY = "unify"

    @staticmethod
    def unify(outlying_scores: np.array) -> np.array:
        # =======================================================================================================================
        # Turn output into probability
        # Source: https://github.com/yzhao062/pyod/blob/8fafcbd7c441954b82db27bfd9cd7f0720974d9a/pyod/models/base.py
        # Reference:
        #   Hans-Peter Kriegel, etc. Interpreting and unifying outlier scores. SDM 2011
        # =======================================================================================================================
        mu = np.mean(outlying_scores)
        sigma = np.std(outlying_scores)
        if sigma !=0:
            pre_erf_score = (outlying_scores - mu) / (sigma * np.sqrt(2))
        else:
            pre_erf_score = (outlying_scores - mu)
        erf_score = erf(pre_erf_score)
        return erf_score.clip(0, 1).ravel()

    @staticmethod
    def zscore(outlying_scores: np.array) -> np.array:
        # =========================================================
        # Turn output into density z-score
        # Reference:
        #   N. X. Vinh etc.
        #   Discovering Outlying Aspects in Large Datasets. DMKD
        # Positive large z-score indicates anomaly
        # =========================================================
        mean = np.mean(outlying_scores)
        std = np.std(outlying_scores)
        if std != 0:
            return (outlying_scores - mean) / std
        else:
            return outlying_scores - mean

class PathManager:
    def __init__(self):
        self.dataset = "../dataset"
        self.output = "output"
        self.batch_model_test = f"{self.output}/batchtest"
        self.debug = "debug"
        if os.path.isdir(self.output) == False:
            os.mkdir(self.output)
        if os.path.isdir(self.debug) == False:
            os.mkdir(self.debug)

    def get_output(self, dataset, sample_method, base_method, aggregate):
        return f"{self.output}/{dataset}_{sample_method}_{base_method}_{aggregate}.json"

    def get_raw_score(self, dataset, sample_method, base_method, aggregate, start_dim, end_dim, ensemble_size):
        return f"{self.debug}/{dataset}_{sample_method}_{base_method}_{aggregate}_{start_dim}_{end_dim}_{ensemble_size}.json"

    def get_batch_test_model_output(self, model_name, aggregator, base_method, normalizer):
        return f"{self.output}/batchtest/{model_name}_{base_method}_{aggregator}_{normalizer}.json"
