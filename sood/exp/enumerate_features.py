#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

from itertools import combinations

import numpy as np
from pyod.utils import precision_n_scores
from sklearn.metrics import roc_auc_score
from sood.util import Normalize
from sood.model.abs_model import Aggregator
from sood.data_process.data_loader import DataLoader, Dataset
from sood.model.base_detectors import kNN, GKE, kNN_GPU
from sood.log import getLogger

logger = getLogger(__name__)

for dataset in [Dataset.BREASTW, Dataset.VERTEBRAL, Dataset.ANNTHYROID,
                Dataset.GLASS, Dataset.PIMA, Dataset.THYROID, ]:
    logger.info("=" * 50)
    logger.info(f"             Dataset {dataset}             ")
    logger.info("=" * 50)
    X, Y = DataLoader.load(dataset)

    model_outputs = []
    total_feature = X.shape[1]
    feature_index = np.array([i for i in range(total_feature)])

    neigh = max(10, int(np.floor(0.03 * X.shape[0])))
    # mdl = kNN(neigh, Normalize.ZSCORE)
    mdl = kNN_GPU(neigh, Normalize.ZSCORE)
    # mdl = GKE(Normalize.ZSCORE)

    for l in range(1, len(feature_index) + 1):
        for i in combinations(feature_index, l):
            selected_features = np.asarray(i)
            _X = X[:, selected_features]
            model_outputs.append(mdl.fit(_X))

    logger.info(f"Total model {len(model_outputs)}")

    count_threshold = 0.2
    score = Aggregator.count_rank_threshold(model_outputs, count_threshold)
    y_scores = np.array(score)
    roc = roc_auc_score(Y, y_scores)
    precision = precision_n_scores(Y, y_scores)
    outlier_subspaces = y_scores[Y == 1]
    inlier_subspaces = y_scores[Y == 0]
    logger.info(f"Outliers: {outlier_subspaces.shape}")
    logger.info(f"Inliers: {inlier_subspaces.shape}")
    logger.info(f"Outlier Subspaces Min: {np.min(outlier_subspaces)} Max: {np.max(outlier_subspaces)} Mean: {np.mean(outlier_subspaces)}")
    logger.info(f"Inlier Subspaces Min: {np.min(inlier_subspaces)} Max: {np.max(inlier_subspaces)} Mean: {np.mean(inlier_subspaces)}")
    logger.info(f"ROC of Count top-{count_threshold} {roc} Precision {precision}")

    count_threshold = 1
    score = Aggregator.count_std_threshold(model_outputs, count_threshold)
    y_scores = np.array(score)
    roc = roc_auc_score(Y, y_scores)
    precision = precision_n_scores(Y, y_scores)
    logger.info(f"ROC of Count {count_threshold}-std {roc} Precision {precision}")

    score = Aggregator.average(model_outputs)
    y_scores = np.array(score)
    roc = roc_auc_score(Y, y_scores)
    precision = precision_n_scores(Y, y_scores)
    logger.info(f"ROC of Average {roc} Precision {precision}")

    average_threshold = 1
    score = Aggregator.average_threshold(model_outputs, average_threshold)
    y_scores = np.array(score)
    roc = roc_auc_score(Y, y_scores)
    precision = precision_n_scores(Y, y_scores)
    logger.info(f"ROC of Average {average_threshold}-std {roc} Precision {precision}")
