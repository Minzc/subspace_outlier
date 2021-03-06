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


def experiment(model, dim_boundary, threshold):
    import json
    path_manager = PathManager()
    ENSEMBLE_SIZE = 100


    if model == "fb":
        Model = FB
    elif model == "oracle":
        Model = OracleAdaptive
    else:
        raise Exception(f"Model not supported {model}")

    threshold = float(threshold)

    for dataset in [Dataset.OPTDIGITS, Dataset.MNIST_ODDS, Dataset.MUSK, Dataset.ARRHYTHMIA,
                    Dataset.AD, Dataset.AID362, Dataset.BANK, Dataset.PROB, Dataset.U2R]:
        for aggregator in [ Aggregator.COUNT_RANK_THRESHOLD, ]:
            for base_model in [kNN.NAME, ]:
                X, Y = DataLoader.load(dataset)
                dim = X.shape[1]
                neigh = max(10, int(np.floor(0.03 * X.shape[0])))
                logger.info(f"{dataset} {aggregator} {threshold}")
                roc_aucs = []
                precision_at_ns = []
                # =======================================================================================
                # Model
                if dim_boundary == "high":
                    start_dim = dim / 2
                    end_dim = dim
                else:
                    start_dim = 2
                    end_dim = dim / 4
                mdl = Model(start_dim, end_dim, ENSEMBLE_SIZE, aggregator, neigh, base_model, Y, threshold)
                output_path = path_manager.get_batch_test_model_output(Model.NAME, aggregator, base_model,
                                                                       "DEFAULT", dataset, start_dim, end_dim)
                logger.info(f"Output File {output_path}")
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
                    Consts.START_DIM: start_dim,
                    Consts.END_DIM: end_dim,
                    Consts.ENSEMBLE_SIZE: ENSEMBLE_SIZE
                }
                with open(output_path, "w") as w:
                    w.write(f"{json.dumps(output)}\n")
                    logger.info(f"Output file is {output_path}")


if __name__ == '__main__':
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", required=["fb", "oracle"])
    parser.add_argument("-d", required=["low", "high"])
    parser.add_argument("-t")
    parsedArgs = parser.parse_args(sys.argv[1:])
    experiment(parsedArgs.m, parsedArgs.d, parsedArgs.t)
