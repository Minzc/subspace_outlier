#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from sood.model.abs_model import AbstractModel, Aggregator

from sood.log import getLogger
from sklearn.neighbors import LocalOutlierFactor
# ====================================================
# Feature Bagging
# - Uniform sampling low dimensional data
# ====================================================

logger = getLogger(__name__)

class FB(AbstractModel):
    def __init__(self, dim_start, dim_end, ensemble_size):
        super().__init__(f"FB({dim_start}-{dim_end}))")
        self.dim_start = dim_start
        self.dim_end = dim_end
        self.ensemble_size = ensemble_size
        self.lof = LocalOutlierFactor(n_neighbors=5, metric="euclidean")

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
            # TODO run outlier detection on the dataset
            self.lof.fit(_X)
            model_outputs.append(self.lof.negative_outlier_factor_)
            logger.debug(f"Outlier score shape: {self.lof.negative_outlier_factor_.shape}")
        return model_outputs

    def aggregate_components(self, model_outputs):
        return Aggregator.count_rank_threshold(model_outputs, 100)

if __name__ == '__main__':
    from sood.data_process.data_loader import Dataset, DataLoader
    X, Y = DataLoader.load(Dataset.ARRHYTHMIA)
    fb = FB(1, 10, 2)
    rst = fb.run(X)
    logger.debug(f"Ensemble output {rst}")