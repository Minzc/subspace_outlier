#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

from sood.model.base_detectors import kNN_LSCP
from sood.data_process.data_loader import DataLoader, Dataset
from sood.log import getLogger
from sood.model.abs_model import AbstractModel
from pyod.models.lscp import LSCP
import numpy as np

logger = getLogger(__name__)


# ======================
# Reference:
#   LSCP: Locally Selective Combination of Parallel Outlier Ensembles. SDM 2019
# Source: https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.lscp
# ======================

class Lscp(AbstractModel):
    NAME = "LSCP"

    def __init__(self, dim_start, dim_end, ensemble_size, neighbor):
        name = f"{self.NAME}({dim_start}-{dim_end} Neighbor: {neighbor}))"
        super().__init__(name, aggregate_method=None, base_model=None, neighbor=neighbor)
        self.dim_start = dim_start
        self.dim_end = dim_end
        self.ensemble_size = ensemble_size
        self.neighbor = neighbor
        np.random.seed(1)

    def compute_ensemble_components(self, data_array):
        detector_list = []
        feature_index = np.array([i for i in range(data_array.shape[1])])
        for i in range(self.ensemble_size):
            # Randomly sample feature size
            feature_size = np.random.randint(self.dim_start, self.dim_end)
            # Randomly select features
            selected_features = np.random.choice(feature_index, feature_size)
            detector_list.append(kNN_LSCP(neighbor_size=self.neighbor, selected_features=selected_features))

        clf = LSCP(detector_list)
        clf.fit(data_array)
        score = clf.decision_scores_
        return [score, ]

    def aggregate_components(self, model_outputs):
        return model_outputs[0]


if __name__ == '__main__':
    X, Y = DataLoader.load(Dataset.OPTDIGITS)
    dim = X.shape[1]
    neigh = max(10, int(np.floor(0.03 * X.shape[0])))
    ENSEMBLE_SIZE = 100

    mdl = Lscp(1, dim / 2, ENSEMBLE_SIZE, neigh)
    rst = mdl.run(X)
    roc_auc = mdl.compute_roc_auc(rst, Y)
    print(f"Final ROC {roc_auc}")
