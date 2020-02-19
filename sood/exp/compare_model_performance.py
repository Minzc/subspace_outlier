#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

from sood.model.weak_oracle_adaptive import OracleAdaptive

from sood.log import getLogger
from sood.model.abs_model import Aggregator
from sood.data_process.data_loader import Dataset, DataLoader
from sood.model.base_detectors import kNN
from sood.model.fb import FB
from sood.util import PathManager, Consts
import numpy as np
import tqdm

logger = getLogger(__name__)


def experiment(model):
    import json
    path_manager = PathManager()
    ENSEMBLE_SIZE = 100

    if model == "fb":
        Model = FB
    elif model == "oracle":
        Model = OracleAdaptive
    else:
        raise Exception(f"Model not supported {model}")

    for dataset in [Dataset.ARRHYTHMIA, Dataset.MUSK, Dataset.MNIST_ODDS, Dataset.OPTDIGITS]:
        for aggregator in [Aggregator.AVERAGE, Aggregator.AVERAGE_THRESHOLD, Aggregator.COUNT_STD_THRESHOLD,
                           Aggregator.COUNT_RANK_THRESHOLD]:
            for base_model in [kNN.NAME, ]:
                # =======================================================================================
                # Model
                output_path = path_manager.get_batch_test_model_output(Model.NAME, aggregator, base_model,
                                                                       "DEFAULT", dataset)
                # =======================================================================================
                with open(output_path, "w") as w:
                    for threshold in [0, ]:
                        X, Y = DataLoader.load(dataset)
                        dim = X.shape[1]
                        neigh = max(10, int(np.floor(0.03 * X.shape[0])))
                        logger.info(f"{dataset} {aggregator} {threshold}")
                        roc_aucs = []
                        precision_at_ns = []
                        # =======================================================================================
                        # Model
                        mdl = Model(2, dim / 4, ENSEMBLE_SIZE, aggregator, neigh, base_model, Y,
                                    threshold)
                        # =======================================================================================
                        for _ in tqdm.trange(5):
                            try:
                                rst = mdl.run(X)
                                # Throw exception if no satisfied subspaces are found
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
                        output = {
                            Consts.DATA: dataset,
                            Consts.ROC_AUC: np.mean(roc_aucs),
                            Consts.PRECISION_A_N: np.mean(precision_at_ns),
                            Consts.AGGREGATE: aggregator,
                            Consts.BASE_MODEL: base_model,
                            Consts.START_DIM: 2,
                            Consts.END_DIM: 1 / 4,
                            Consts.ENSEMBLE_SIZE: ENSEMBLE_SIZE
                        }
                        w.write(f"{json.dumps(output)}\n")
                        logger.info(f"Output file is {output_path}")


if __name__ == '__main__':
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "model", required=["fb", "oracle"])
    parsedArgs = parser.parse_args(sys.argv[1:])
    experiment(parsedArgs.m)
