#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

import tqdm
from sood.data_process.data_loader import DataLoader, Dataset
from sood.log import getLogger
from sood.model.abs_model import AbstractModel, Aggregator
from sood.util import Similarity, PathManager, Consts, Normalize
import numpy as np

from sood.model.base_detectors import kNN

logger = getLogger(__name__)


def calculate_weights(global_rank_list, local_rank_list, selected_features, total_feature):
    # s_score = Similarity.pearson(global_rank_list, local_rank_list, if_weighted=False)  # Compute spearman correlation
    s_score = Similarity.cosine(global_rank_list, local_rank_list)  # Compute spearman correlation
    logger.info(f"Correlation {s_score}")
    # s_score = Similarity.spearman(global_rank_list, local_rank_list)  # Compute spearman correlation
    choice_probability = [1] * total_feature
    for f_idx in selected_features:
        choice_probability[f_idx] += s_score
    choice_probability = np.exp(choice_probability)
    normalizer = sum(choice_probability)
    choice_probability = choice_probability / normalizer
    return choice_probability


def get_global_rank_list(global_rank_list, model_outputs):
    # weights = [Similarity.pearson(global_rank_list, i, if_weighted=False) for i in model_outputs]
    if len(model_outputs) > 0:
        weights = [Similarity.cosine(global_rank_list, i) for i in model_outputs]
        # weights = [Similarity.spearman(global_rank_list, i) for i in model_outputs]
        weights = [i if i > 0 else 0 for i in weights]
        normalize = sum(weights)
        if normalize > 0:
            weights = [i / normalize for i in weights]
        global_rank_list = np.array(Aggregator.average(model_outputs, weights)) * len(model_outputs)
    return global_rank_list


class OracleAdaptive(AbstractModel):
    NAME = "OracleAdaptive"

    def __init__(self, dim_start, dim_end, ensemble_size, aggregate_method, neighbor, base_model, Y, threshold):
        name = f"{self.NAME}({dim_start}-{dim_end} Neighbor: {neighbor}))"
        super().__init__(name, aggregate_method, base_model, neighbor, norm_method=Normalize.ZSCORE)
        self.dim_start = dim_start
        self.dim_end = dim_end
        self.ensemble_size = ensemble_size
        self.aggregate_method = aggregate_method
        self.Y = Y
        self.threshold = threshold
        np.random.seed(1)

    def compute_ensemble_components(self, data_array):
        model_outputs = []
        total_feature = data_array.shape[1]
        feature_index = np.array([i for i in range(total_feature)])
        selected_features = feature_index
        rocs = []
        local_rank_list = Aggregator.average_threshold([self.mdl.fit(data_array), ], 2)

        global_rank_list = local_rank_list

        for i in range(self.ensemble_size):
            logger.info(f"Select features {selected_features}")

            choice_probability = calculate_weights(global_rank_list, local_rank_list, selected_features, total_feature)

            # ============================================================
            # 1. Randomly sample feature size
            # 2. Randomly select features
            # ============================================================
            feature_size = np.random.randint(self.dim_start, self.dim_end)
            selected_features = np.random.choice(feature_index, feature_size, p=choice_probability, replace=False)
            # selected_features = np.random.choice(feature_index, feature_size, replace=False)

            local_rank_list = self.mdl.fit(data_array[:, selected_features])
            if np.sum(local_rank_list) == 0:
                continue
            global_rank_list = get_global_rank_list(global_rank_list, model_outputs)

            # ==============================
            # DEBUG
            roc_auc = self.compute_roc_auc(global_rank_list, self.Y)
            logger.info("Ensemble Before {}".format(roc_auc))
            logger.info(f"Precision@n {self.compute_precision_at_n(global_rank_list, self.Y)}")
            local_roc = self.compute_roc_auc(np.array(self.aggregate_components([local_rank_list, ])), self.Y)
            local_precision = self.compute_precision_at_n(np.array(self.aggregate_components([local_rank_list, ])),
                                                          self.Y)
            logger.info(f"Local {local_roc}")
            logger.info(f"Local Precision {local_precision}")
            if local_roc < 0.5:
                Similarity.cosine(global_rank_list, local_rank_list, self.Y)
            rocs.append(local_roc)
            # ==============================

            model_outputs.append(local_rank_list)

            # ==============================
            # DEBUG
            # s_score = Similarity.pearson(global_rank_list, local_rank_list, False, self.Y)
            global_rank_list = get_global_rank_list(global_rank_list, model_outputs)
            roc_auc = self.compute_roc_auc(global_rank_list, self.Y)
            logger.info("Ensemble After {}".format(roc_auc))
            logger.info(f"Precision@n {self.compute_precision_at_n(global_rank_list, self.Y)}")
            logger.info('-' * 50)
            # ==============================
        logger.info("Number of good subspace {}/{}".format(len(rocs), self.ensemble_size))
        logger.info("Maximum roc {}".format(max(rocs)))
        logger.info("Minimum roc {}".format(min(rocs)))
        return model_outputs

    def aggregate_components(self, model_outputs):
        if self.aggregate_method == Aggregator.COUNT_RANK_THRESHOLD:
            return Aggregator.count_rank_threshold(model_outputs, 0.2)
        elif self.aggregate_method == Aggregator.AVERAGE:
            return Aggregator.average(model_outputs)
        elif self.aggregate_method == Aggregator.COUNT_STD_THRESHOLD:
            return Aggregator.count_std_threshold(model_outputs, 1)
        elif self.aggregate_method == Aggregator.AVERAGE_THRESHOLD:
            return Aggregator.average_threshold(model_outputs, 1)


def single_test():
    dataset = Dataset.MNIST_ODDS
    aggregator = Aggregator.AVERAGE_THRESHOLD
    threshold = 0

    X, Y = DataLoader.load(dataset)
    dim = X.shape[1]
    neigh = max(10, int(np.floor(0.03 * X.shape[0])))
    ENSEMBLE_SIZE = 100
    logger.info(f"{dataset} {aggregator} {threshold}")
    roc_aucs = []
    precision_at_ns = []
    mdl = OracleAdaptive(2, dim / 4, ENSEMBLE_SIZE, aggregator, neigh, kNN.NAME, Y, threshold)
    for _ in tqdm.trange(1):
        try:
            rst = mdl.run(X)
            roc_auc = mdl.compute_roc_auc(rst, Y)
            logger.info("Final ROC {}".format(roc_auc))
            precision_at_n = mdl.compute_precision_at_n(rst, Y)
            logger.info("Precision@n {}".format(precision_at_n))

            roc_aucs.append(roc_auc)
            precision_at_ns.append(precision_at_n)
        except Exception as e:
            logger.exception(e)
    logger.info(f"Exp Information {dataset} {aggregator} {threshold}")
    logger.info(f"Final Average ROC {np.mean(roc_aucs)}")
    logger.info(f"Final Precision@n {np.mean(precision_at_ns)}")
    logger.info(f"====================================================")


if __name__ == '__main__':
    single_test()
