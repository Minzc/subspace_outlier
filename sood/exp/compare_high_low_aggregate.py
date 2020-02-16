#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
import time
import json
import numpy as np
from sood.model.fb import FB
from sood.model.base_detectors import kNN, LOF
from sood.data_process.data_loader import Dataset, DataLoader
from sood.model.abs_model import Aggregator
from sood.log import getLogger
from sood.util import PathManager

logger = getLogger(__name__)

EXP_NUM = 20

class ExpConfig:
    def __init__(self, dataset, aggregate, base_model, neighbor, ensemble_size, start, end, X, Y):
        self.dataset = dataset
        self.aggregate = aggregate
        self.base_model = base_model
        self.neighbor = neighbor
        self.ensemble_size = ensemble_size
        self.start_dim = start
        self.end_dim = end
        self.X = X
        self.Y = Y
        self.EXP_NUM = EXP_NUM

    def to_json(self):
        return {
            "dataset": self.dataset,
            "aggregate": self.aggregate,
            "base_model": self.base_model,
            "neighbor": self.neighbor,
            "ensemble_size": self.ensemble_size,
            "start_dim": self.start_dim,
            "end_dim": self.end_dim
        }


def generate_exp_conditions():
    for dataset in Dataset.supported_dataset():
        for aggregate in Aggregator.supported_aggregate():
            for base_model in [kNN.NAME, LOF.NAME]:
                X, Y = DataLoader.load(Dataset.ARRHYTHMIA)
                dim = X.shape[1]
                neighbor = max(10, int(np.floor(0.03 * X.shape[0])))
                for ensemble_size in [10, 50, 100, 150, 200, 250, 300]:
                    for start, end in [(4 * int(dim / 10), 5 * int(dim / 10)),
                                       (3 * int(dim / 10), 4 * int(dim / 10)),
                                       (2 * int(dim / 10), 3 * int(dim / 10)),
                                       (1 * int(dim / 10), 2 * int(dim / 10)),
                                       (1, int(dim / 10)),
                                       (1, int(dim / 2)),
                                       (int(dim / 2), dim)]:
                        yield ExpConfig(dataset,
                                        aggregate,
                                        base_model,
                                        neighbor,
                                        ensemble_size,
                                        start,
                                        end,
                                        X, Y)


def exp(exp_config: ExpConfig):
    fb = FB(exp_config.start_dim, exp_config.end_dim,
            exp_config.ensemble_size, exp_config.aggregate,
            exp_config.neighbor, exp_config.base_model)

    start_ts = time.time()
    roc_aucs = []

    for i in range(exp_config.EXP_NUM):
        logger.info(f"Start running {exp_config.to_json()} Iteration: {i}")
        rst = fb.run(exp_config.X)
        roc_auc = fb.compute_roc_auc(rst, exp_config.Y)
        roc_aucs.append(roc_auc)

    end_ts = time.time()
    logger.info(f"""
    Model: {exp_config.to_json()} 
    ROC AUC {np.mean(roc_aucs)} 
    Std: {np.std(roc_aucs)} 
    Time Elapse: {end_ts - start_ts}""")
    return roc_aucs, end_ts - start_ts

def main():
    path_manager = PathManager()
    for exp_config in generate_exp_conditions():
        with open(path_manager.get_output(exp_config.dataset, "uniform", exp_config.base_model), "a") as w:
            roc_aucs, elapse_time = exp(exp_config)
            result = exp_config.to_json()
            result["roc_aucs"] = roc_aucs
            result["time"] = elapse_time
            w.write(f"{json.dumps(result)}\n")


