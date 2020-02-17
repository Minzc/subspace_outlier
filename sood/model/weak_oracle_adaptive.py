#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

from sood.data_process.data_loader import DataLoader, Dataset
from pyod.utils.stat_models import wpearsonr
from sood.log import getLogger
from scipy.stats import spearmanr
from sood.model.abs_model import AbstractModel, Aggregator
import numpy as np

logger = getLogger(__name__)


def jaccard(g_array, l_array):
    g_array = g_array[:150]
    l_array = l_array[:150]
    g_set = set(g_array.tolist())
    l_set = set(l_array.tolist())
    return len(g_set & l_set) / len(g_set | l_set)

def spearman(g_array, l_array):
    g_rank = [0] * g_array.shape[0]
    l_rank = [0] * l_array.shape[0]

    for rank, idx in enumerate(g_array):
        g_rank[idx] = rank
    for rank, idx in enumerate(l_array):
        l_rank[idx] = rank
    return spearmanr(g_rank, l_rank)[0]

def pearson(initial_rank_list, local_rank_list):
    global_outlying_idx = np.argsort(initial_rank_list)[::-1]
    local_outlying_idx = np.argsort(local_rank_list)[::-1]
    weights = [0] * initial_rank_list.shape[0]
    for rank, idx in enumerate(global_outlying_idx):
        weights[idx] = 1/(rank + 1)
    logger.info(f"Score {initial_rank_list[global_outlying_idx[0]]} {initial_rank_list[global_outlying_idx[1]]}")
    return wpearsonr(initial_rank_list, local_outlying_idx, weights)


class OracleAdaptive(AbstractModel):
    NAME = "OracleAdaptive"

    def __init__(self, dim_start, dim_end, ensemble_size, aggregate_method, neighbor, base_model, Y):
        name = f"{self.NAME}({dim_start}-{dim_end} Neighbor: {neighbor}))"
        super().__init__(name, aggregate_method, base_model, neighbor)
        self.dim_start = dim_start
        self.dim_end = dim_end
        self.ensemble_size = ensemble_size
        self.aggregate_method = aggregate_method
        np.random.seed(1)
        self.Y = Y

    def compute_ensemble_components(self, data_array):
        model_outputs = []
        total_feature = data_array.shape[1]
        feature_index = np.array([i for i in range(total_feature)])
        initial_count = int(self.ensemble_size * 0.1)
        selected_features = None
        local_rank_list = None
        counter = 0

        for i in range(initial_count):
            counter += 1
            # Randomly sample feature size
            feature_size = np.random.randint(self.dim_start, self.dim_end)
            # Randomly select features
            selected_features = np.random.choice(feature_index, feature_size)
            _X = data_array[:, selected_features]
            # Process selected dataset
            local_rank_list = self.mdl.fit(_X)
            model_outputs.append(local_rank_list)
            logger.debug(f"Outlier score shape: {local_rank_list.shape}")

        global_rank_list = np.array(self.aggregate_components(model_outputs))
        for i in range(initial_count, self.ensemble_size):
            counter += 1
            # j_score = jaccard(global_outlying_idx, local_outlying_idx)
            s_score = pearson(global_rank_list, local_rank_list)

            choice_probability = [1] * total_feature
            for f_idx in selected_features:
                choice_probability[f_idx] += s_score
            choice_probability = np.exp(choice_probability)
            # choice_probability[f_idx] += j_score

            normalizer = sum(choice_probability)
            for i in range(total_feature):
                choice_probability[i] /= normalizer
            logger.info(f"Correlation {s_score}")

            # Randomly sample feature size
            feature_size = np.random.randint(self.dim_start, self.dim_end)
            # Randomly select features
            selected_features = np.random.choice(feature_index, feature_size, p=choice_probability)
            # selected_features = np.random.choice(feature_index, feature_size)
            _X = data_array[:, selected_features]
            # Process selected dataset
            local_rank_list = self.mdl.fit(_X)
            model_outputs.append(local_rank_list)
            global_rank_list = np.array(self.aggregate_components(model_outputs))
            roc_auc = mdl.compute_roc_auc(global_rank_list, Y)
            print(f"Ensemble {roc_auc}")

            local_roc = mdl.compute_roc_auc(np.array(self.aggregate_components([local_rank_list, ])), Y)
            print(f"Local {local_roc}")
        print("Subspace", counter)
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
    from sood.model.base_detectors import kNN

    X, Y = DataLoader.load(Dataset.OPTDIGITS)
    dim = X.shape[1]
    neigh = max(10, int(np.floor(0.03 * X.shape[0])))
    ENSEMBLE_SIZE = 100

    mdl = OracleAdaptive(1, dim / 2, ENSEMBLE_SIZE, Aggregator.AVERAGE, neigh, kNN.NAME, Y)
    rst = mdl.run(X)
    roc_auc = mdl.compute_roc_auc(rst, Y)
    print(f"Final ROC {roc_auc}")
