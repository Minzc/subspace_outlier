#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.metrics import roc_auc_score
from sood.log import getLogger
import numpy as np

logger = getLogger(__name__)


class Aggregator:
    COUNT_RANK_THRESHOLD = "count_rank_threshold"
    AVERAGE = "average"

    @staticmethod
    def count_rank_threshold(model_outputs, threshold):
        # =================================
        # Small value means outlying
        # =================================
        scores = [0] * model_outputs[0].shape[0]
        logger.debug(f"Score size {len(scores)}")
        for model_output in model_outputs:
            outlying_idx = np.argsort(model_output)[::-1][:threshold]
            for idx in outlying_idx:
                # logger.debug(f"Idx {idx} Score {model_output[idx]}")
                scores[idx] += 1
        return scores

    @staticmethod
    def average(model_outputs):
        # =================================
        # Small value means outlying
        # =================================
        scores = [0] * model_outputs[0].shape[0]
        logger.debug(f"Score size {len(scores)}")
        for model_output in model_outputs:
            for idx, score in enumerate(model_output):
                scores[idx] += score
        for i in range(len(scores)):
            scores[i] = scores[i] / len(model_outputs)
        return scores


class AbstractModel:
    def __init__(self, name, aggregate_method):
        self.name = name
        self.if_normalize_score = True
        if aggregate_method == Aggregator.COUNT_RANK_THRESHOLD:
            self.if_normalize_score = False

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


if __name__ == '__main__':
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([-10, -7, -7.5, -3])
    print(roc_auc_score(y_true, y_scores))
