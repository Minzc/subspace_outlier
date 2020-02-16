#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from sood.model.base_detectors import LOF
import time
from sood.model.abs_model import AbstractModel, Aggregator
from sood.log import getLogger

# ====================================================
# Feature Bagging
# - Uniform sampling low dimensional data
# ====================================================

logger = getLogger(__name__)


class FB(AbstractModel):
    def __init__(self, dim_start, dim_end, ensemble_size, aggregate_method):
        super().__init__(f"FB({dim_start}-{dim_end}))", aggregate_method)
        self.dim_start = dim_start
        self.dim_end = dim_end
        self.ensemble_size = ensemble_size
        self.aggregate_method = aggregate_method
        self.lof = LOF(10, self.if_normalize_score)


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
            score = self.lof.fit(_X)
            model_outputs.append(score)
            logger.debug(f"Outlier score shape: {score.shape}")
        return model_outputs

    def aggregate_components(self, model_outputs):
        return Aggregator.count_rank_threshold(model_outputs, 100)
        # return Aggregator.average(model_outputs)


if __name__ == '__main__':
    from sood.data_process.data_loader import Dataset, DataLoader

    X, Y = DataLoader.load(Dataset.ARRHYTHMIA)
    for start, end in [(1, 140), (1, 120), (1, 100), (1, 80), (1, 60), (1, 40), (1, 20), (100, 274)]:
        start_ts = time.time()
        fb = FB(start, end, 200, Aggregator.COUNT_RANK_THRESHOLD)
        rst = fb.run(X)
        logger.debug(f"Ensemble output {rst}")
        logger.debug(f"Y {Y}")
        total = 0
        exp_num = 1000
        for i in range(exp_num):
            roc_auc = fb.compute_roc_auc(rst, Y)
            total += roc_auc
        end_ts = time.time()
        logger.info(f"Model: {fb.info()} ROC AUC {total / exp_num} Time Elapse: {end_ts - start_ts}")

    logger.info("=" * 50)
    logger.info("Finish")
