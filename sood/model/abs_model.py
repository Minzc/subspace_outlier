#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.metrics import roc_auc_score
from sood.model.base_detectors import kNN, LOF
from sood.log import getLogger
from pyod.utils.utility import precision_n_scores
import numpy as np

logger = getLogger(__name__)


class Aggregator:
    COUNT_RANK_THRESHOLD = "count_rank_threshold"
    AVERAGE = "average"
    AVERAGE_THRESHOLD = "average_threshold"
    COUNT_STD_THRESHOLD = "count_std_threshold"

    @classmethod
    def supported_aggregate(cls):
        return [cls.AVERAGE, cls.COUNT_RANK_THRESHOLD, cls.COUNT_STD_THRESHOLD, cls.AVERAGE_THRESHOLD]

    @staticmethod
    def count_std_threshold(model_outputs, threshold):
        # =================================
        # Small value means outlying
        # =================================
        scores = [0] * model_outputs[0].shape[0]
        logger.debug(f"Score size {len(scores)}")
        for model_output in model_outputs:
            mean = np.mean(model_output)
            std = np.std(model_output)
            t = mean + threshold * std
            outlying_idx = model_output > t

            for idx, if_outlying in enumerate(outlying_idx):
                if if_outlying:
                    logger.debug(f"Idx {idx} Score {model_output[idx]} T: {t} if_outlying: {if_outlying}")
                    logger.debug(f"Mean {mean} Std {std}")
                    scores[idx] += 1
        return scores

    @staticmethod
    def count_rank_threshold(model_outputs, threshold):
        # =================================
        # Large value means outlying
        # =================================
        scores = [0] * model_outputs[0].shape[0]
        threshold = max(int(len(scores) * threshold) + 1, 10)
        logger.debug(f"Score size {len(scores)} Threshold {threshold}")
        for model_output in model_outputs:
            # np.argsort() Sort in ascending order
            outlying_idx = np.argsort(model_output)[::-1][:threshold]
            logger.debug(f"Score {model_output[outlying_idx[0]]} {model_output[outlying_idx[1]]}")
            for idx in outlying_idx:
                scores[idx] += 1
        return scores

    @staticmethod
    def average(model_outputs):
        # =================================
        # Large value means outlying
        # =================================
        scores = [0] * model_outputs[0].shape[0]
        logger.debug(f"Score size {len(scores)}")
        for model_output in model_outputs:
            for idx, score in enumerate(model_output):
                assert np.isnan(scores[idx] + score) == False, (scores[idx], score, model_output)
                scores[idx] += score

        for i in range(len(scores)):
            scores[i] = scores[i] / len(model_outputs)

        return scores

    @staticmethod
    def average_threshold(model_outputs, M):
        # =================================
        # Small value means outlying
        # =================================
        scores = [0] * model_outputs[0].shape[0]
        logger.debug(f"Score size {len(scores)}")
        for model_output in model_outputs:
            for idx, score in enumerate(model_output):
                assert np.isnan(scores[idx] + score) == False, (scores[idx], score, model_output)
                if score >= M:
                    scores[idx] += score

        for i in range(len(scores)):
            scores[i] = scores[i] / len(model_outputs)

        return scores


class AbstractModel:
    def __init__(self, name, aggregate_method, base_model, neighbor):
        self.name = name
        self.if_normalize_score = False
        if aggregate_method == Aggregator.AVERAGE:
            self.if_normalize_score = True
        elif aggregate_method == Aggregator.AVERAGE_THRESHOLD:
            self.if_normalize_score = True

        if base_model == kNN.NAME:
            self.mdl = kNN(neighbor, self.if_normalize_score)
        elif base_model == LOF.NAME:
            self.mdl = LOF(neighbor, self.if_normalize_score)
        else:
            raise Exception(f"Base Model: {base_model} is not supported.")

    def info(self):
        return f"{self.name} IF_NORMALIZE_SCORE: {self.if_normalize_score}"

    def compute_ensemble_components(self, data_array):
        pass

    def aggregate_components(self, model_outputs):
        pass

    def run(self, data_array):
        model_outputs = self.compute_ensemble_components(data_array)
        rst = self.aggregate_components(model_outputs)
        return rst

    def compute_roc_auc(self, rst, ground_truth):
        y_scores = np.array(rst)
        return roc_auc_score(ground_truth, y_scores)

    def compute_precision_at_n(self, rst, ground_truth, n=None):
        y_scores = np.array(rst)
        return precision_n_scores(ground_truth, y_scores, n)


if __name__ == '__main__':
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([-10, -7, -7.5, -3])
    print(roc_auc_score(y_true, y_scores))
