#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

from collections import defaultdict
from itertools import combinations
import numpy as np
import matplotlib
from sood.util import Normalize
from sood.model.abs_model import Aggregator
from sood.data_process.data_loader import DataLoader, Dataset
from sood.model.base_detectors import kNN, GKE_GPU
from sood.log import getLogger

logger = getLogger(__name__)


def outliers_per_subspace():
    import json
    outputs = defaultdict(dict)
    model = 'gke'

    for dataset in [Dataset.GLASS, Dataset.WINE,
                    Dataset.BREASTW, Dataset.ANNTHYROID,
                    Dataset.VOWELS, Dataset.PIMA, Dataset.THYROID]:
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
        selected_features = []
        for l in range(1, len(feature_index) + 1):
            for i in combinations(feature_index, l):
                model_outputs.append( (i, mdl.fit(X_gpu_tensor[:, np.asarray(i)])) )

        logger.info(f"Total model {len(model_outputs)}")
        for name, aggregator, threshold in [("RANK", Aggregator.count_rank_threshold, 0.05),
                                            ("RANK", Aggregator.count_rank_threshold, 0.10),
                                            ("STD", Aggregator.count_std_threshold, 1),
                                            ("STD", Aggregator.count_std_threshold, 2)]:
            logger.info(f"---------------{name}------------------------")
            outlier_num_per_subspace = []
            for selected_features, i in model_outputs:
                y_scores = np.array(aggregator([i, ], threshold))

                outlier_num_per_subspace.append(int(np.sum(y_scores[Y == 1])))
            outputs[f"{name}_{threshold}"][dataset] = {
                "outlier_dist": outlier_num_per_subspace,
                "outlier_total": int(outlier_num),
                "subspace_total": len(model_outputs)
            }

        total_score = Aggregator.count_rank_threshold(model_outputs)
        for idx, i in enumerate(Y):
            if i  == 1 and total_score[i] == 0:
                print("FN Outliers", X_gpu_tensor[idx])
        print("Inliers", X_gpu_tensor[Y == 0])

    output_file = f"{model}_outliers_per_subspace.json"
    with open(output_file, "w") as w:
        w.write(f"{json.dumps(outputs)}\n")
    logger.info(f"Output file {output_file}")


if __name__ == '__main__':
    outliers_per_subspace()
