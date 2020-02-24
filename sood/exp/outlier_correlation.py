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
        subspace_idx = []
        for l in range(1, len(feature_index) + 1):
            for i in combinations(feature_index, l):
                model_outputs.append(mdl.fit(X_gpu_tensor[:, np.asarray(i)]))
                subspace_idx.append(i)

        logger.info(f"Total model {len(model_outputs)}")
        for name, aggregator, threshold in [("RANK", Aggregator.count_rank_threshold, 0.05),
                                            ("RANK", Aggregator.count_rank_threshold, 0.10),
                                            ("STD", Aggregator.count_std_threshold, 1),
                                            ("STD", Aggregator.count_std_threshold, 2)]:
            outlier_idx = {idx: [] for idx, y in enumerate(Y) if y == 1}
            outliers_per_subspace = {idx: aggregator([i, ], threshold) for idx, i in enumerate(model_outputs)}
            for subspace_idx, outliers in outliers_per_subspace.items():
                for outlier in outliers:
                    outlier_idx[outlier].append(subspace_idx)
            not_covered_outliers = {i for i, subspaces in outlier_idx.items() if len(subspaces) > 0}
            logger.info(f"Detected outliers {len(not_covered_outliers)}")



    # output_file = f"{model}_outliers_correlation_subspace.json"
    # with open(output_file, "w") as w:
    #     w.write(f"{json.dumps(outputs)}\n")
    # logger.info(f"Output file {output_file}")


if __name__ == '__main__':
    outlier_correlation_subspace()
