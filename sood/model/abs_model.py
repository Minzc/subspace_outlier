#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
from collections import defaultdict
import numpy as np
from sood.log import getLogger

logger = getLogger(__name__)

class Aggregator:
    @staticmethod
    def count_rank_threshold(model_outputs, threshold):
        # =================================
        # Small value means outlying
        # =================================
        scores = [0] * model_outputs[0].shape[0]
        logger.debug(f"Score size {len(scores)}")
        for model_output in model_outputs:
            outlying_idx = np.argsort(model_output)[:threshold]
            for idx in outlying_idx:
                # logger.debug(f"Idx {idx} Score {model_output[idx]}")
                scores[idx] += 1
        return scores

class AbstractModel:
    def __init__(self, name):
        self.name = name

    def compute_ensemble_components(self, data_array):
        pass

    def aggregate_components(self, model_outputs):
        pass

    def run(self, data_array):
        model_outputs = self.compute_ensemble_components(data_array)
        rst = self.aggregate_components(model_outputs)
        return rst