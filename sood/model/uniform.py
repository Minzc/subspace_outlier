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


class Uniform(AbstractModel):
    NAME = "UNIFORM"
    def __init__(self, dim_start, dim_end, ensemble_size, aggregate_method, neighbor, base_model):
        name = f"UNIFORM({dim_start}-{dim_end} Neighbor: {neighbor}))"
        super().__init__(name, aggregate_method, base_model, neighbor)
        self.dim_start = dim_start
        self.dim_end = dim_end
        self.ensemble_size = ensemble_size
        self.aggregate_method = aggregate_method
        np.random.seed(1)

    def compute_ensemble_components(self, data_array):
        model_outputs = []
        total_features = data_array.shape[1]
        feature_index = np.array([i for i in range(total_features)])
        for i in range(self.ensemble_size):
            # Randomly sample feature size
            feature_size = np.random.randint(self.dim_start, self.dim_end)
            # Randomly select features
            selected_features = np.random.choice(feature_index, len(feature_index), replace=False)[:feature_size]
            _X = data_array[:, selected_features]
            logger.info(f"Selected X: {_X.shape}")
            # Process selected dataset
            score = self.mdl.fit(_X)
            model_outputs.append(score)
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
    EXP_NUM = 1

    X, Y = DataLoader.load(Dataset.MNIST_ODDS)
    neigh = max(10, int(np.floor(0.03 * X.shape[0])))
    # X = X[:, np.std(X, axis=0) != 0]
    dim = X.shape[1]

    for start, end in [(2, int(dim / 4))]:
        fb = Uniform(start, end, ENSEMBLE_SIZE, Aggregator.AVERAGE, neigh, kNN.NAME)

        start_ts = time.time()
        roc_aucs = []
        precision_at_ns = []

        for i in range(EXP_NUM):
            rst = fb.run(X)

            logger.info(f"Ensemble output {rst}")
            logger.info(f"Y {Y}")

            roc_auc = fb.compute_roc_auc(rst, Y)
            roc_aucs.append(roc_auc)

            precision_at_n = fb.compute_precision_at_n(rst, Y)
            precision_at_ns.append(precision_at_n)

        end_ts = time.time()
        logger.info(
            f""" Model: {fb.info()} ROC AUC {np.mean(roc_aucs)} Std: {np.std(roc_aucs)} Precision@n {np.mean(precision_at_ns)} Std: {np.std(precision_at_ns)} Time Elapse: {end_ts - start_ts}""")

    logger.info("=" * 50)
