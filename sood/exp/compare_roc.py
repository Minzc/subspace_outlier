#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

from collections import defaultdict
from itertools import combinations
import time
import numpy as np
import matplotlib
import json

matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pyod.utils import precision_n_scores
from sklearn.metrics import roc_auc_score
from sood.util import Normalize
from sood.model.abs_model import Aggregator
from sood.data_process.data_loader import DataLoader, Dataset
from sood.model.base_detectors import kNN, GKE_GPU
from sood.log import getLogger

logger = getLogger(__name__)


def compare_auc():
    outputs = defaultdict(dict)
    model_name = "knn"
    # model_name = "gke"
    for dataset in [Dataset.VOWELS, Dataset.WINE,
                    Dataset.BREASTW, Dataset.ANNTHYROID,
                    Dataset.GLASS, Dataset.PIMA, Dataset.THYROID]:
        logger.info("=" * 50)
        logger.info(f"             Dataset {dataset}             ")
        logger.info("=" * 50)
        _X, Y = DataLoader.load(dataset)
        feature_index = np.array([i for i in range(_X.shape[1])])

        if model_name == "knn":
            X_gpu_tensor = _X
            mdl = kNN(max(10, int(np.floor(0.03 * _X.shape[0]))), Normalize.ZSCORE)
        elif model_name == "gke":
            X_gpu_tensor = GKE_GPU.convert_to_tensor(_X)
            mdl = GKE_GPU(Normalize.ZSCORE)

        model_outputs = []
        for l in range(1, len(feature_index) + 1):
            for i in combinations(feature_index, l):
                model_outputs.append(mdl.fit(X_gpu_tensor[:, np.asarray(i)]))

        full_roc = roc_auc_score(Y, np.array(mdl.fit(X_gpu_tensor)))
        full_precision = precision_n_scores(Y, np.array(mdl.fit(X_gpu_tensor)))

        logger.info(f"Total model {len(model_outputs)}")
        for name, aggregator, threshold in [ ("AVG", Aggregator.average, None),
                                            ("AVG", Aggregator.average_threshold, 1),
                                            ("AVG", Aggregator.average_threshold, 2),
                                            ]:
            if threshold is not None:
                y_scores = np.array(aggregator(model_outputs, threshold))
            else:
                y_scores = np.array(aggregator(model_outputs))
            roc = roc_auc_score(Y, y_scores)
            precision = precision_n_scores(Y, y_scores)
            logger.info(f"ROC of {name}-{threshold} {roc} Precision {precision}")
            outputs[dataset][f"{name}_{threshold}"] = {
                "roc": roc,
                "precision": precision,
                "full_roc": full_roc,
                "full_precision": full_precision
            }

    output_file = f"{model_name}_performance.json"
    with open(output_file, "w") as w:
        w.write(f"{json.dumps(outputs)}\n")
    logger.info(f"Output file {output_file}")
    print(outputs)

if __name__ == '__main__':
    compare_auc()

