#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

from typing import List

import numpy as np
from pyod.utils import wpearsonr
from scipy.stats import spearmanr

from sood.model.base_detectors import LOF, kNN
import time
from sood.model.abs_model import AbstractModel, Aggregator
from sood.log import getLogger
from sood.util import Similarity

# ====================================================
# Select Ensemble Models
# Reference:
#   S Rayana. etc. Less is More: Building Selective Anomaly Ensembles. ICDM 2015
# ====================================================
from sood.util import Normalize

logger = getLogger(__name__)


class SelectV(AbstractModel):
    NAME = "SelectV"

    def __init__(self, dim_start, dim_end, ensemble_size, neighbor, base_model):
        name = f"FB({dim_start}-{dim_end} Neighbor: {neighbor}))"
        super().__init__(name, Aggregator.AVERAGE, base_model, neighbor, norm_method=None)
        self.dim_start = dim_start
        self.dim_end = dim_end
        self.ensemble_size = ensemble_size
        self.aggregate_method = Aggregator.AVERAGE
        np.random.seed(1)


    def average_model_outputs(self, model_output_probas: List) -> np.array:
        # ==================================================
        # Average the probability score across lists
        # ==================================================
        target = np.zeros(model_output_probas[0].shape)
        for i in range(len(model_output_probas)):
            target += model_output_probas[i]
        target = target / len(model_output_probas)
        return target

    def model_selection(self, S: List) -> List:
        P = [Normalize.unify(i) for i in S] # Shape (instance, )
        target = self.average_model_outputs(P)

        # Sort P by their weighted pearson to target in descending order
        P = list(sorted(P, key=lambda i: Similarity.pearson(target, i, if_weighted=True), reverse=True))
        assert Similarity.pearson(target, P[0], True) >= Similarity.pearson(target, P[1], True),\
            f"Pearson to target {Similarity.pearson(target, P[0], True)} {Similarity.pearson(target, P[1], True)}"
        E = [P.pop(0), ]


        while len(P) > 0:
            # Current prediction of E
            p = self.average_model_outputs(E)
            # Sort P by wP correlation to p in descending order
            P = list(sorted(P, key=lambda i: Similarity.pearson(p, i, True), reverse=True))
            if len(P) > 2:
                assert Similarity.pearson(p, P[0], True) >= Similarity.pearson(p, P[1], True), \
                    f"Pearson to p {Similarity.pearson(p, P[0], True)} {Similarity.pearson(p, P[1], True)}"
            l = P.pop(0)
            sim_p_target = Similarity.pearson(target, p, True)
            sim_E_union_l_target = Similarity.pearson(target, self.average_model_outputs(E + [l, ]), True)
            # Select list if the correlation improved by this addition
            if sim_E_union_l_target > sim_p_target:
                E.append(l)
        logger.info(f"Number of selected models {len(E)}")
        return E

    def compute_ensemble_components(self, data_array):
        model_outputs = []
        feature_index = np.array([i for i in range(data_array.shape[1])])
        for i in range(self.ensemble_size):
            # Randomly sample feature size
            feature_size = np.random.randint(self.dim_start, self.dim_end)
            # Randomly select features
            selected_features = np.random.choice(feature_index, feature_size)
            logger.debug(f"Feature size: {feature_size} Selected X: {data_array[:, selected_features].shape}")
            logger.debug(f"Selected feature: {selected_features}")
            # Process selected dataset
            score = self.mdl.fit(data_array[:, selected_features])
            model_outputs.append(score)
            logger.debug(f"Outlier score shape: {score.shape}")
        return self.model_selection(model_outputs)

    def aggregate_components(self, model_outputs):
        return Aggregator.average(model_outputs)


if __name__ == '__main__':
    from sood.data_process.data_loader import Dataset, DataLoader

    ENSEMBLE_SIZE = 100
    EXP_NUM = 1
    PRECISION_AT_N = 10

    X, Y = DataLoader.load(Dataset.OPTDIGITS)
    dim = X.shape[1]
    neigh = max(10, int(np.floor(0.03 * X.shape[0])))

    # for start, end in [ (1, int(dim / 2)), (int(dim / 2), dim)]:
    for start, end in [(1, int(dim / 2)), ]:
        selectv = SelectV(start, end, ENSEMBLE_SIZE, neigh, kNN.NAME)

        start_ts = time.time()
        roc_aucs = []
        precision_at_ns = []

        for i in range(EXP_NUM):
            rst = selectv.run(X)

            logger.debug(f"Ensemble output {rst[:10]}")
            logger.debug(f"Y {Y[:10]}")

            roc_auc = selectv.compute_roc_auc(rst, Y)
            roc_aucs.append(roc_auc)

            precision_at_n = selectv.compute_precision_at_n(rst, Y, PRECISION_AT_N)
            precision_at_ns.append(precision_at_n)

        end_ts = time.time()
        logger.info(
            f""" Model: {selectv.info()} ROC AUC {np.mean(roc_aucs)} Std: {np.std(roc_aucs)} Precision@n {np.mean(precision_at_ns)} Std: {np.std(precision_at_ns)} Time Elapse: {end_ts - start_ts}""")