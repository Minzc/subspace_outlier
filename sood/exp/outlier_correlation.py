#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
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
        outlier_num = np.sum(Y == 1)
        feature_index = np.array(range(_X.shape[1]))
        if model == "knn":
            mdl = kNN(max(10, int(np.floor(0.03 * _X.shape[0]))), Normalize.ZSCORE)
            X_gpu_tensor = _X
        elif model == "gke":
            mdl = GKE_GPU(Normalize.ZSCORE)
            X_gpu_tensor = GKE_GPU.convert_to_tensor(_X)

        model_outputs = []
        subspace_idx_to_feautres = []
        for l in range(1, len(feature_index) + 1):
            for i in combinations(feature_index, l):
                model_outputs.append(mdl.fit(X_gpu_tensor[:, np.asarray(i)]))
                subspace_idx_to_feautres.append([int(j) for j in i])

        logger.info(f"Total model {len(model_outputs)}")
        for name, aggregator, threshold in [("RANK", Aggregator.count_rank_threshold, 0.05),
                                            ("RANK", Aggregator.count_rank_threshold, 0.10),
                                            ("STD", Aggregator.count_std_threshold, 1),
                                            ("STD", Aggregator.count_std_threshold, 2)]:
            outliers_to_subspaces = defaultdict(set)
            subspace_to_outlier = {}
            for subspace_id, model_output in enumerate(model_outputs):
                detected_outliers = {
                    point_idx
                    for point_idx, if_outlier in enumerate(aggregator([model_output, ], threshold))
                    if if_outlier == 1 and Y[point_idx] == 1
                }
                subspace_to_outlier[subspace_id] = detected_outliers
                for detected_outlier in detected_outliers:
                    outliers_to_subspaces[detected_outlier].add(subspace_id)
            _subspace_to_outlier = {i: copy.deepcopy(j) for i, j in subspace_to_outlier.items()}

            not_covered_outliers = {i for i, subspaces in outliers_to_subspaces.items() if len(subspaces) > 0}
            not_covered_outliers_num = len(not_covered_outliers)
            logger.info(f"Detected outliers {len(not_covered_outliers)}/{outlier_num}")
            selected_subspaces = []
            while len(not_covered_outliers) > 0:
                _tmp = sorted(subspace_to_outlier.items(), key=lambda x: len(x[1]), reverse=True)
                selected_subspace_id, covered_outliers = \
                    sorted(subspace_to_outlier.items(), key=lambda x: len(x[1]), reverse=True)[0]
                not_covered_outliers = not_covered_outliers - covered_outliers
                subspace_to_outlier = {i: (j - covered_outliers) for i, j in subspace_to_outlier.items()}
                selected_subspaces.append(selected_subspace_id)

            for i in selected_subspaces:
                print(f"Features {subspace_idx_to_feautres[i]} Outliers {len(_subspace_to_outlier[i])}")
            print(f"{len(selected_subspaces)}/{len(model_outputs)}")
            outputs[f"{name}_{threshold}"][dataset] = {
                "select_subspace": [(subspace_idx_to_feautres[i], list(_subspace_to_outlier[i])) for i in
                                    selected_subspaces],
                "outliers": not_covered_outliers_num,
                "total_subspace": len(model_outputs),
                "total_outliers": outlier_num,
                "dimension": len(feature_index)
            }

    output_file = f"{model}_outliers_correlation_subspace.json"
    with open(output_file, "w") as w:
        w.write(f"{json.dumps(outputs)}\n")
    logger.info(f"Output file {output_file}")


def plot_correlation():
    model = "knn"
    input_file = f"{model}_outliers_correlation_subspace.json"
    with open(input_file) as f:
        obj = json.loads(f.readlines()[0])

    for aggregator_threshold, data in obj.items():
        aggregator, threshold = aggregator_threshold.split("_")
        print("=" * 20)
        print(f"{aggregator} {threshold}")
        print("=" * 20)
        for dataset, exp_rst in data.items():
            jaccards = []
            for i, j in itertools.combinations(exp_rst['select_subspaces'], 2):
                jaccards.append(
                    len(set(i[1]) & set(j[1])) / len(set(i[1]) | set(j[1]))
                )
            print(
                f"{dataset} & {exp_rst['outliers']}/{exp_rst['total_outliers']} & {len(exp_rst['select_subspace'])} & {np.mean(jaccards)}/{np.min(jaccards)}/{np.max(jaccards)}")


if __name__ == '__main__':
    outlier_correlation_subspace()
    plot_correlation()
