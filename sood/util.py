#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

import os


class Consts:
    DATA = "data"
    LABEL = "label"
    ROC_AUC = "roc_aucs"
    PRECISION_A_N = "precision@n"
    TIME = "time"

    DATA_SET = "dataset"
    AGGREGATE = "aggregate"
    BASE_MODEL = "base_model"
    NEIGHBOR_SIZE = "neighbor_size"
    ENSEMBLE_SIZE = "ensemble_size"
    START_DIM = "start_dim"
    END_DIM = "end_dim"

class PathManager:
    def __init__(self):
        self.dataset = "../dataset"
        self.output = "output"
        self.debug = "debug"
        if os.path.isdir(self.output) == False:
            os.mkdir(self.output)
        if os.path.isdir(self.debug) == False:
            os.mkdir(self.debug)

    def get_output(self, dataset, sample_method, base_method, aggregate):
        return f"{self.output}/{dataset}_{sample_method}_{base_method}_{aggregate}.json"

    def get_raw_score(self, dataset, sample_method, base_method, aggregate, start_dim, end_dim, ensemble_size):
        return f"{self.debug}/{dataset}_{sample_method}_{base_method}_{aggregate}_{start_dim}_{end_dim}_{ensemble_size}.json"