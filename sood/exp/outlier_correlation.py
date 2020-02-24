#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

import itertools
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

def outlier_correlation_subspace():
    import json
    outputs = defaultdict(dict)
    model = 'knn'

    for dataset in [Dataset.VOWELS, Dataset.WINE,
                    Dataset.BREASTW, Dataset.ANNTHYROID,
                    Dataset.GLASS, Dataset.PIMA, Dataset.THYROID]:
        logger.info("=" * 50)
        logger.info(f"             Dataset {dataset}             ")
        logger.info("=" * 50)
        _X, Y = DataLoader.load(dataset)
        outlier_num, inlier_num = np.sum(Y == 1), np.sum(Y == 0)
        feature_index = np.array([i for i in range(_X.shape[1])])
        if model == "knn":
            mdl = kNN(max(10, int(np.floor(0.03 * _X.shape[0]))), Normalize.ZSCORE)
            X_gpu_tensor = _X
        elif model == "gke":
            mdl = GKE_GPU(Normalize.ZSCORE)
            X_gpu_tensor = GKE_GPU.convert_to_tensor(_X)

        model_outputs = []
        for l in range(1, len(feature_index) + 1):
            for i in combinations(feature_index, l):
                model_outputs.append(mdl.fit(X_gpu_tensor[:, np.asarray(i)]))

        logger.info(f"Total model {len(model_outputs)}")
        for name, aggregator, threshold in [("RANK", Aggregator.count_rank_threshold, 0.05),
                                            ("RANK", Aggregator.count_rank_threshold, 0.10),
                                            ("STD", Aggregator.count_std_threshold, 1),
                                            ("STD", Aggregator.count_std_threshold, 2)]:
            outliers_per_subspace = [aggregator([i, ], threshold) for i in model_outputs]
            indexs = np.array(range(len(X_gpu_tensor.shape[0])))
            jaccards = []
            for list_1, list_2 in itertools.combinations(outliers_per_subspace):
                list_1 = set(indexs[list_1 > 0].tolist())
                list_2 = set(indexs[list_2 > 0].tolist())
                jaccard = (list_1 & list_2) / (list_1 | list_2)
                logger.info(f"Instance 1 {len(list_1)}")
                logger.info(f"Instance 2 {len(list_2)}")
                logger.info(f"Jaccard {jaccard}")
                jaccards.append(jaccard)


            outlier_num_per_subspace = []
            for i in model_outputs:
                y_scores = np.array(aggregator([i, ], threshold))
                outlier_num_per_subspace.append(int(np.sum(y_scores[Y == 1])))
            outputs[f"{name}_{threshold}"][dataset] = {
                "jaccard_dist": jaccards
            }

    output_file = f"{model}_outliers_correlation_subspace.json"
    with open(output_file, "w") as w:
        w.write(f"{json.dumps(outputs)}\n")
    logger.info(f"Output file {output_file}")

if __name__ == '__main__':
    outlier_correlation_subspace()
