#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from pyod.utils import standardizer

from sood.model.base_detectors import LOF, kNN

from sood.model.abs_model import AbstractModel, Aggregator

from sood.log import getLogger
from sklearn.neighbors import LocalOutlierFactor

# ====================================================
# Feature Bagging
# - Uniform sampling low dimensional data
# ====================================================

logger = getLogger(__name__)


class Full(AbstractModel):
    def __init__(self, neighbor, base_model):
        super().__init__(f"FULL", Aggregator.AVERAGE, base_model, neighbor)

    def compute_ensemble_components(self, data_array):
        model_outputs = []
        # score = self.lof.fit(data_array)
        score = self.mdl.fit(data_array)
        model_outputs.append(score)
        logger.debug(f"Outlier score shape: {score.shape}")

        return model_outputs

    def aggregate_components(self, model_outputs):
        return Aggregator.average(model_outputs)


if __name__ == '__main__':
    from sood.data_process.data_loader import Dataset, DataLoader
    import scipy.io as sio
    X, Y = DataLoader.load(Dataset.SPEECH)

    # data = sio.loadmat("data/musk.mat")
    # X_train = data['X'].astype('double')
    # y_label = np.squeeze(data['y']).astype('int')
    neigh = max(10, int(np.floor(0.03 * X.shape[0])))

    # neigh = max(10, int(np.floor(0.03 * X.shape[0])))
    full = Full(neigh, kNN.NAME)
    rst = full.run(X)
    logger.debug(f"Ensemble output {rst}")
    roc_auc = full.compute_roc_auc(rst, Y)
    logger.info(f"ROC AUC {roc_auc}")
    precision_at_n = full.compute_precision_at_n(rst, Y)
    logger.info(f"Precision at N {precision_at_n}")
