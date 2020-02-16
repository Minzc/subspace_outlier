#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from sood.model.base_detectors import LOF, kNN
import time
from sood.model.abs_model import AbstractModel, Aggregator
from sood.log import getLogger

# ====================================================
# Feature Bagging
# - Uniform sampling low dimensional data
# ====================================================

logger = getLogger(__name__)


class FB(AbstractModel):
    def __init__(self, dim_start, dim_end, ensemble_size, aggregate_method, neighbor, base_model):
        name = f"FB({dim_start}-{dim_end} Neighbor: {neighbor}))"
        super().__init__(name, aggregate_method, base_model, neighbor)
        self.dim_start = dim_start
        self.dim_end = dim_end
        self.ensemble_size = ensemble_size
        self.aggregate_method = aggregate_method
        np.random.seed(1)

    def compute_ensemble_components(self, data_array):
        model_outputs = []
        feature_index = np.array([i for i in range(data_array.shape[1])])
        for i in range(self.ensemble_size):
            feature_size = np.random.randint(self.dim_start, self.dim_end)
            selected_features = np.random.choice(feature_index, feature_size)
            logger.debug(f"Feature size: {feature_size}")
            logger.debug(f"Selected feature: {selected_features}")
            _X = data_array[:, selected_features]
            logger.debug(f"Selected X: {_X.shape}")
            # Process selected dataset
            score = self.mdl.fit(_X)
            model_outputs.append(score)
            logger.debug(f"Outlier score shape: {score.shape}")
        return model_outputs

    def aggregate_components(self, model_outputs):
        if self.aggregate_method == Aggregator.COUNT_RANK_THRESHOLD:
            return Aggregator.count_rank_threshold(model_outputs, 100)
        elif self.aggregate_method == Aggregator.AVERAGE:
            return Aggregator.average(model_outputs)
        elif self.aggregate_method == Aggregator.COUNT_STD_THRESHOLD:
            return Aggregator.count_std_threshold(model_outputs, 2)
        elif self.aggregate_method == Aggregator.AVERAGE_THRESHOLD:
            return Aggregator.average_threshold(model_outputs, 2)


if __name__ == '__main__':
    from sood.data_process.data_loader import Dataset, DataLoader

    ENSEMBLE_SIZE = 100
    EXP_NUM = 10
    PRECISION_AT_N = 10

    X, Y = DataLoader.load(Dataset.ARRHYTHMIA)
    dim = X.shape[1]
    neigh = max(10, int(np.floor(0.03 * X.shape[0])))

    for start, end in [(4 * int(dim / 10), 5 * int(dim / 10)),
                       (3 * int(dim / 10), 4 * int(dim / 10)),
                       (2 * int(dim / 10), 3 * int(dim / 10)),
                       (1 * int(dim / 10), 2 * int(dim / 10)),
                       (1, int(dim / 10)),
                       (1, int(dim / 2)),
                       (int(dim / 2), dim)]:
        fb = FB(start, end, ENSEMBLE_SIZE, Aggregator.AVERAGE_THRESHOLD, neigh, kNN.NAME)

        start_ts = time.time()
        roc_aucs = []
        precision_at_ns = []

        for i in range(EXP_NUM):
            rst = fb.run(X)

            logger.debug(f"Ensemble output {rst}")
            logger.debug(f"Y {Y}")

            roc_auc = fb.compute_roc_auc(rst, Y)
            roc_aucs.append(roc_auc)

            precision_at_n = fb.compute_precision_at_n(rst, Y, PRECISION_AT_N)
            precision_at_ns.append(precision_at_n)

        end_ts = time.time()
        logger.info(
            f""" Model: {fb.info()} ROC AUC {np.mean(roc_aucs)} Std: {np.std(roc_aucs)} Precision@n {np.mean(precision_at_ns)} Std: {np.std(precision_at_ns)} Time Elapse: {end_ts - start_ts}""")

    logger.info("=" * 50)
    for start, end in [(4 * int(dim / 10), 5 * int(dim / 10)),
                       (3 * int(dim / 10), 4 * int(dim / 10)),
                       (2 * int(dim / 10), 3 * int(dim / 10)),
                       (1, 2 * int(dim / 10)),
                       (1, int(dim / 10)),
                       (1, int(dim / 2)),
                       (int(dim / 2), dim)]:
        fb = FB(start, end, ENSEMBLE_SIZE, Aggregator.AVERAGE, neigh, kNN.NAME)

        start_ts = time.time()
        roc_aucs = []
        precision_at_ns = []

        for i in range(EXP_NUM):
            rst = fb.run(X)

            logger.debug(f"Ensemble output {rst}")
            logger.debug(f"Y {Y}")

            roc_auc = fb.compute_roc_auc(rst, Y)
            roc_aucs.append(roc_auc)

            precision_at_n = fb.compute_precision_at_n(rst, Y, PRECISION_AT_N)
            precision_at_ns.append(precision_at_n)

        end_ts = time.time()
        logger.info(
            f""" Model: {fb.info()} ROC AUC {np.mean(roc_aucs)} Std: {np.std(roc_aucs)} Precision@n {np.mean(precision_at_ns)} Std: {np.std(precision_at_ns)} Time Elapse: {end_ts - start_ts}""")

    logger.info("Finish")
