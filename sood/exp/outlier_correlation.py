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


def load_model_and_data(dataset, model):
    logger.info("=" * 50)
    logger.info(f"             Dataset {dataset}             ")
    logger.info("=" * 50)
    _X, Y = DataLoader.load(dataset)
    outlier_num = int(np.sum(Y == 1))
    feature_index = np.array(range(_X.shape[1]))
    if model == "knn":
        mdl = kNN(max(10, int(np.floor(0.03 * _X.shape[0]))), Normalize.ZSCORE)
        X_gpu_tensor = _X
    elif model == "gke":
        mdl = GKE_GPU(Normalize.ZSCORE)
        X_gpu_tensor = GKE_GPU.convert_to_tensor(_X)
    return mdl, X_gpu_tensor, Y, outlier_num, feature_index


def outlier_subspace_jump_path():
    import json
    outputs = defaultdict(dict)
    model = 'gke'

    for dataset in [Dataset.VOWELS, Dataset.WINE,
                    Dataset.BREASTW, Dataset.ANNTHYROID,
                    Dataset.GLASS, Dataset.PIMA, Dataset.THYROID]:

        mdl, X_gpu_tensor, Y, outlier_num, feature_index = load_model_and_data(dataset, model)

        model_outputs = []
        subspace_idx_to_features = []
        for l in range(1, len(feature_index) + 1):
            for i in combinations(feature_index, l):
                model_outputs.append(mdl.fit(X_gpu_tensor[:, np.asarray(i)]))
                subspace_idx_to_features.append([int(j) for j in i])
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

            covered_outliers = set()
            detected_outliers_num = len({i for i, subspaces in outliers_to_subspaces.items() if len(subspaces) > 0})
            logger.info(f"Detected outliers {detected_outliers_num}/{outlier_num}")
            selected_subspaces = []
            while len(covered_outliers) != detected_outliers_num:
                _tmp = sorted(subspace_to_outlier.items(), key=lambda x: (len(x[1] & covered_outliers) > 0,
                                                                       len(x[1] - covered_outliers)),
                                                        reverse=True)
                _tmp = [(len(x[1] & covered_outliers), len(x[1] - covered_outliers))  for x in _tmp]
                print(_tmp)
                _outliers = set()
                for j in subspace_to_outlier.values():
                    _outliers |= set(j)
                print(_outliers)

                selected_subspace_id, outliers = sorted(subspace_to_outlier.items(),
                                                        key=lambda x: (len(x[1] & covered_outliers) > 0,
                                                                       len(x[1] - covered_outliers)),
                                                        reverse=True)[0]
                print(f"Overlay Subspace {len(outliers & covered_outliers)} {len(covered_outliers)}")
                covered_outliers |= outliers
                selected_subspaces.append(selected_subspace_id)

            for i in selected_subspaces:
                print(f"Features {subspace_idx_to_features[i]} Outliers {len(_subspace_to_outlier[i])}")
            print(f"{len(selected_subspaces)}/{len(model_outputs)}")
            outputs[f"{name}_{threshold}"][dataset] = {
                "select_subspace": [(subspace_idx_to_features[i], list(_subspace_to_outlier[i])) for i in
                                    selected_subspaces],
                "outliers": detected_outliers_num,
                "total_subspace": len(model_outputs),
                "total_outliers": outlier_num,
                "dimension": len(feature_index)
            }

    output_file = f"{model}_outliers_subspace_jump.json"
    with open(output_file, "w") as w:
        w.write(f"{json.dumps(outputs)}\n")
    logger.info(f"Output file {output_file}")


def outlier_correlation_subspace():
    import json
    outputs = defaultdict(dict)
    model = 'gke'

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
                "total_outliers": int(outlier_num),
                "dimension": len(feature_index)
            }

    output_file = f"{model}_outliers_correlation_subspace.json"
    with open(output_file, "w") as w:
        w.write(f"{json.dumps(outputs)}\n")
    logger.info(f"Output file {output_file}")


def plot_correlation():
    model = "gke"
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
            for i, j in itertools.combinations(exp_rst['select_subspace'], 2):
                jaccards.append(
                    len(set(i[1]) & set(j[1])) / len(set(i[1]) | set(j[1]))
                )
            mean_jaccard = np.mean(jaccards) if len(jaccards) > 0 else 0
            min_jaccard = np.min(jaccards) if len(jaccards) > 0 else 0
            max_jaccard = np.max(jaccards) if len(jaccards) > 0 else 0
            total_outliers = exp_rst["total_outliers"]
            covered_outliers = exp_rst['outliers']
            select_subspace_num = len(exp_rst['select_subspace'])
            total_subspaces = exp_rst["total_subspace"]
            print(
                f"{dataset} & {covered_outliers}/{total_outliers} & {select_subspace_num}/{total_subspaces} & {mean_jaccard:.2f}/{min_jaccard:.2f}/{max_jaccard:.2f} \\\\n")
            print("\hline")


if __name__ == '__main__':
    # outlier_correlation_subspace()
    # plot_correlation()
    outlier_subspace_jump_path()
