#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from sood.model.base_detectors import LOF

from sood.model.abs_model import AbstractModel, Aggregator

from sood.log import getLogger
from sklearn.neighbors import LocalOutlierFactor
# ====================================================
# Feature Bagging
# - Uniform sampling low dimensional data
# ====================================================

logger = getLogger(__name__)

class Full(AbstractModel):
    def __init__(self):
        super().__init__(f"FULL")
        self.lof = LOF(10, False)

    def compute_ensemble_components(self, data_array):
        model_outputs = []
        score = self.lof.fit(data_array)
        model_outputs.append(score)
        logger.debug(f"Outlier score shape: {score.shape}")

        return model_outputs

    def aggregate_components(self, model_outputs):
        return Aggregator.average(model_outputs)

if __name__ == '__main__':
    from sood.data_process.data_loader import Dataset, DataLoader
    X, Y = DataLoader.load(Dataset.ARRHYTHMIA)
    full = Full()
    rst = full.run(X)
    logger.debug(f"Ensemble output {rst}")
    roc_auc = full.compute_roc_auc(rst, Y)
    logger.info(f"ROC AUC {roc_auc}")