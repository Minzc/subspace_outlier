#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
import time
import json
import numpy as np
import tqdm

from sood.model.fb import FB
from sood.model.base_detectors import kNN, LOF
from sood.data_process.data_loader import Dataset, DataLoader
from sood.model.abs_model import Aggregator
from sood.log import getLogger
from sood.model.uniform import Uniform
from sood.util import PathManager, Consts

logger = getLogger(__name__)

EXP_NUM = 20
ENSEMBLE_SIZES = [100, ]


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
            Consts.DATA_SET: self.dataset,
            Consts.AGGREGATE: self.aggregate,
            Consts.BASE_MODEL: self.base_model,
            Consts.NEIGHBOR_SIZE: self.neighbor,
            Consts.ENSEMBLE_SIZE: self.ensemble_size,
            Consts.START_DIM: self.start_dim,
            Consts.END_DIM: self.end_dim
        }


def generate_exp_conditions():
    for dataset in Dataset.supported_dataset():
        for aggregate in Aggregator.supported_aggregate():
            base_model = kNN.NAME
            X, Y = DataLoader.load(dataset)
            dim = X.shape[1]
            neighbor = max(10, int(np.floor(0.03 * X.shape[0])))
            for ensemble_size in ENSEMBLE_SIZES:
                for start, end in [(1, int(dim / 2)), (int(dim / 2), dim)]:
                    yield ExpConfig(dataset,
                                    aggregate,
                                    base_model,
                                    neighbor,
                                    ensemble_size,
                                    start,
                                    end,
                                    X, Y)
                logger.info("=" * 50)


def exp(exp_config: ExpConfig, path_manager: PathManager, Model):
    mdl = Model(exp_config.start_dim, exp_config.end_dim,
            exp_config.ensemble_size, exp_config.aggregate,
            exp_config.neighbor, exp_config.base_model)

    start_ts = time.time()
    roc_aucs = []
    precision_at_ns = []

    with open(path_manager.get_raw_score(exp_config.dataset, mdl.NAME, exp_config.base_model, exp_config.aggregate,
                                         exp_config.start_dim, exp_config.end_dim, exp_config.ensemble_size), "w") as w:
        logger.info(f"Start running {Model.NAME} {exp_config.to_json()}")
        for _ in tqdm.trange(exp_config.EXP_NUM):
            rst = mdl.run(exp_config.X)
            roc_auc = mdl.compute_roc_auc(rst, exp_config.Y)
            roc_aucs.append(roc_auc)
            precision_at_n = mdl.compute_precision_at_n(rst, exp_config.Y)
            precision_at_ns.append(precision_at_n)

            w.write(f"{json.dumps(rst)}\n")

    end_ts = time.time()
    logger.info(f"Sampling Method {Model.NAME}")
    logger.info(f"""Avg. ROC AUC {np.mean(roc_aucs)} 
    Avg. Precision@m {np.mean(precision_at_ns)} 
    Std. ROC AUC: {np.std(roc_aucs)}
    Std. Precision@m: {np.std(precision_at_ns)} 
    Time Elapse: {end_ts - start_ts}""".replace("\n", " "))
    return roc_aucs, precision_at_ns, end_ts - start_ts


def main():
    path_manager = PathManager()
    for exp_config in generate_exp_conditions():
        for Model in [FB, Uniform]:
            with open(path_manager.get_output(exp_config.dataset, Model.NAME, exp_config.base_model, exp_config.aggregate), "w") as w:
                roc_aucs, precision_at_ns, elapse_time = exp(exp_config, path_manager, Model)
                result = exp_config.to_json()
                result[Consts.ROC_AUC] = (np.mean(roc_aucs), np.std(roc_aucs))
                result[Consts.PRECISION_A_N] = (np.mean(precision_at_ns), np.std(precision_at_ns))
                result[Consts.TIME] = elapse_time
                w.write(f"{json.dumps(result)}\n")
            logger.info("-" * 50)

if __name__ == '__main__':
    main()
