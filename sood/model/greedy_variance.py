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
        weights[idx] = 1 / (rank + 1)
    logger.info(f"Score {initial_rank_list[global_outlying_idx[0]]} {initial_rank_list[global_outlying_idx[1]]}")
    return wpearsonr(initial_rank_list, local_outlying_idx, weights)


class GreedyVariance(AbstractModel):
    NAME = "GreedyVariance"

    def __init__(self, dim_start, dim_end, ensemble_size, aggregate_method, neighbor, base_model, Y, threshold):
        name = f"{self.NAME}({dim_start}-{dim_end} Neighbor: {neighbor}))"
        super().__init__(name, aggregate_method, base_model, neighbor)
        self.dim_start = dim_start
        self.dim_end = dim_end
        self.ensemble_size = ensemble_size
        self.aggregate_method = aggregate_method
        self.threshold = threshold
        np.random.seed(1)
        self.Y = Y

    def compute_ensemble_components(self, data_array):
        model_outputs = []
        total_feature = data_array.shape[1]
        feature_index = np.array([i for i in range(total_feature)])
        feature_w = np.std(data_array, axis=0)
        logger.info("STD : {}".format(feature_w.shape))
        feature_w = np.exp(feature_w)
        normalizer = sum(feature_w)
        feature_w = feature_w / normalizer
        logger.info("Feature weight : {}".format(feature_w.shape))
        counter = 0
        rocs = []

        for i in range(self.ensemble_size):
            # Randomly sample feature size
            feature_size = np.random.randint(self.dim_start, self.dim_end)
            # Randomly select features
            selected_features = np.random.choice(feature_index, feature_size)
            _X = data_array[:, selected_features]
            # Process selected dataset
            local_rank_list = self.mdl.fit(_X)

            if len(model_outputs):
                roc_auc = mdl.compute_roc_auc(np.array(self.aggregate_components(model_outputs)), Y)
                print("Ensemble Before {}".format(roc_auc))

            local_roc = mdl.compute_roc_auc(np.array(self.aggregate_components([local_rank_list, ])), Y)
            print(f"Local {local_roc}")
            rocs.append(local_roc)
            if local_roc > self.threshold:
                counter += 1
                model_outputs.append(local_rank_list)
                roc_auc = mdl.compute_roc_auc(np.array(self.aggregate_components(model_outputs)), Y)
                print("Ensemble After {}".format(roc_auc))
            print('-' * 50)
        logger.info("Number of good subspace {}/{}".format(counter, self.ensemble_size))
        logger.info("Maixmum roc {}".format(max(rocs)))
        logger.info("Minimum roc {}".format(min(rocs)))
        return model_outputs

    def aggregate_components(self, model_outputs):
        if self.aggregate_method == Aggregator.COUNT_RANK_THRESHOLD:
            return Aggregator.count_rank_threshold(model_outputs, 0.05)
        elif self.aggregate_method == Aggregator.AVERAGE:
            return Aggregator.average(model_outputs)
        elif self.aggregate_method == Aggregator.COUNT_STD_THRESHOLD:
            return Aggregator.count_std_threshold(model_outputs, 2)
        elif self.aggregate_method == Aggregator.AVERAGE_THRESHOLD:
            return Aggregator.average_threshold(model_outputs, 2)


def test():
    for dataset in [Dataset.ARRHYTHMIA, Dataset.OPTDIGITS, Dataset.MUSK, Dataset.MNIST_ODDS]:
        for aggregator in [Aggregator.COUNT_STD_THRESHOLD, Aggregator.AVERAGE_THRESHOLD]:
            for threshold in [0, 0.5, 0.7, 0.9]:
                X, Y = DataLoader.load(dataset)
                dim = X.shape[1]
                neigh = max(10, int(np.floor(0.03 * X.shape[0])))
                ENSEMBLE_SIZE = 100
                logger.info(f"{dataset} {aggregator} {threshold}")
                mdl = GreedyVariance(1, dim / 2, ENSEMBLE_SIZE, aggregator, neigh, kNN.NAME, Y, threshold)
                try:
                    rst = mdl.run(X)
                    roc_auc = mdl.compute_roc_auc(rst, Y)
                    logger.info("Final ROC {}".format(roc_auc))
                    precision_at_n = mdl.compute_precision_at_n(rst, Y)
                    logger.info("Precision@n {}".format(precision_at_n))
                except Exception as e:
                    logger.exception(e)

if __name__ == '__main__':
    from sood.model.base_detectors import kNN
    X, Y = DataLoader.load(Dataset.OPTDIGITS)
    dim = X.shape[1]
    neigh = max(10, int(np.floor(0.03 * X.shape[0])))
    ENSEMBLE_SIZE = 100
    mdl = GreedyVariance(1, dim / 2, ENSEMBLE_SIZE, Aggregator.AVERAGE, neigh, kNN.NAME, Y, 0)
    try:
        rst = mdl.run(X)
        roc_auc = mdl.compute_roc_auc(rst, Y)
        logger.info("Final ROC {}".format(roc_auc))
        precision_at_n = mdl.compute_precision_at_n(rst, Y)
        logger.info("Precision@n {}".format(precision_at_n))
    except Exception as e:
        logger.exception(e)