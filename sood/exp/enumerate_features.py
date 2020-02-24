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
from sood.model.base_detectors import kNN#, GKE_GPU
from sood.log import getLogger

logger = getLogger(__name__)


def compare_auc():
    outputs = defaultdict(dict)
    # model = "knn"
    model_name = "gke"
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

        logger.info(f"Total model {len(model_outputs)}")
        for name, aggregator, threshold in [("RANK", Aggregator.count_rank_threshold, 0.05),
                                            ("RANK", Aggregator.count_rank_threshold, 0.10),
                                            ("STD", Aggregator.count_std_threshold, 1),
                                            ("STD", Aggregator.count_std_threshold, 2),
                                            ("AVG", Aggregator.average, None),
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
                "precision": precision
            }

    output_file = f"{model_name}_performance.json"
    with open(output_file, "w") as w:
        w.write(f"{json.dumps(outputs)}\n")
    logger.info(f"Output file {output_file}")

def outliers_per_subspace():
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
            outlier_num_per_subspace = []
            for i in model_outputs:
                y_scores = np.array(aggregator([i, ], threshold))
                outlier_num_per_subspace.append(int(np.sum(y_scores[Y == 1])))
            outputs[f"{name}_{threshold}"][dataset] = {
                "outlier_dist": outlier_num_per_subspace,
                "outlier_total": int(outlier_num),
                "subspace_total": len(model_outputs)
            }


    output_file = f"{model}_outliers_per_subspace.json"
    with open(output_file, "w") as w:
        w.write(f"{json.dumps(outputs)}\n")
    logger.info(f"Output file {output_file}")



def point_count_per_dim():
    import json
    outputs = defaultdict(dict)
    model = "knn"
    for dataset in [Dataset.VOWELS, Dataset.WINE,
                    Dataset.BREASTW, Dataset.ANNTHYROID,
                    Dataset.GLASS, Dataset.PIMA, Dataset.THYROID]:
        logger.info("=" * 50)
        logger.info(f"             Dataset {dataset}             ")
        logger.info("=" * 50)
        _X, Y = DataLoader.load(dataset)
        feature_index = np.array([i for i in range(_X.shape[1])])
        outlier_num, inlier_num = np.sum(Y == 1), np.sum(Y == 0)

        if model == "knn":
            mdl = GKE_GPU(Normalize.ZSCORE)
            X_gpu_tensor = GKE_GPU.convert_to_tensor(_X)
        else:
            mdl = kNN(max(10, int(np.floor(0.03 * _X.shape[0]))), Normalize.ZSCORE)
            X_gpu_tensor = _X


        model_outputs_all = defaultdict(list)
        for l in range(1, len(feature_index) + 1):
            for i in combinations(feature_index, l):
                model_outputs_all[l].append(mdl.fit(X_gpu_tensor[:, np.asarray(i)]))

        assert len(model_outputs_all) == len(feature_index)

        for name, aggregator, threshold in [("RANK", Aggregator.count_rank_threshold, 0.05),
                                            ("RANK", Aggregator.count_rank_threshold, 0.10),
                                            ("STD", Aggregator.count_std_threshold, 1),
                                            ("STD", Aggregator.count_std_threshold, 2)]:
            dim_outlier_ratio = [0] * len(feature_index)
            dim_inlier_ratio = [0] * len(feature_index)

            for l, model_outputs in model_outputs_all.items():
                y_scores = np.array(aggregator(model_outputs, threshold))

                point_idx = set()
                for idx, score in enumerate(y_scores[Y == 1]):
                    if score > 0:
                        point_idx.add(idx)
                dim_outlier_ratio[l - 1] = len(point_idx) / outlier_num

                point_idx = set()
                for idx, score in enumerate(y_scores[Y == 0]):
                    if score > 0:
                        point_idx.add(idx)
                dim_inlier_ratio[l - 1] = len(point_idx) / inlier_num

            outputs[f"{name}_{threshold}"][dataset] = {
                "outlier": dim_outlier_ratio,
                "inlier": dim_inlier_ratio,
                "feature_index": feature_index.tolist()
            }

    with open(f"{model}_point_count_per_dim.json", "w") as w:
        w.write(f"{json.dumps(outputs)}")


def autolabel(ax, rects, digits=1):
    """Attach a text label above each bar in *rects*, displaying its height."""
    counter = 0
    max_height = max([rect.get_height() for rect in rects])
    for rect in rects:
        counter += 1
        height = rect.get_height()
        if height == max_height or counter == len(rects):
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', color='red')


def subspace_count_per_point():
    import json
    BIN_NUM = 10
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
            y_scores = np.array(aggregator(model_outputs, threshold))
            outlier_subspaces, inlier_subspaces = y_scores[Y == 1], y_scores[Y == 0]

            outlier_hist, bin = np.histogram(outlier_subspaces, BIN_NUM, range=(0.1, len(model_outputs)))
            bin = [f"{i / len(model_outputs):.1f}" for i in bin]
            zero_subspaces_outlier = sum([1 for i in outlier_subspaces if i == 0])
            print(zero_subspaces_outlier)
            print(outlier_hist)
            outlier_hist = np.insert(outlier_hist, 0, zero_subspaces_outlier)
            print(outlier_hist)
            assert np.sum(outlier_hist) == outlier_num

            inlier_hist = np.histogram(inlier_subspaces, BIN_NUM, range=(0.1, len(model_outputs)))[0]
            zero_subspaces_inlier = sum([1 for i in inlier_subspaces if i == 0])
            print(zero_subspaces_inlier)
            print(inlier_hist)
            inlier_hist = np.insert(inlier_hist, 0, zero_subspaces_inlier)
            print(inlier_hist)
            assert np.sum(inlier_hist) == inlier_num

            outlier_hist_percent = outlier_hist / outlier_num
            inlier_hist_percent = inlier_hist / inlier_num

            logger.info(f"Outlier {outlier_num} Inlier {inlier_num}")
            logger.info(f"Outlier Median {np.median(outlier_subspaces)} Inlier Median {np.median(inlier_subspaces)}")
            logger.info(f"Outlier Mean {np.mean(outlier_subspaces)} Inlier Mean {np.mean(inlier_subspaces)}")

            logger.info(f"Bin {bin}")
            logger.info(f"Outlier dist {outlier_hist}")
            logger.info(f"Inlier dist {inlier_hist}")
            logger.info(f"Outlier dist density {outlier_hist_percent}")
            logger.info(f"Inlier dist density {inlier_hist_percent}")

            outputs[f"{name}_{threshold}"][dataset] = {
                "outlier": outlier_hist_percent.tolist(),
                "inlier": inlier_hist_percent.tolist(),
                "bin": bin,
                "outlier_mean": np.mean(outlier_subspaces),
                "inlier_mean": np.mean(inlier_subspaces),
                "outlier_median": np.median(outlier_subspaces),
                "inlier_median": np.median(inlier_subspaces),
            }

    output_file = f"{model}_subspace_count_per_point.json"
    with open(output_file, "w") as w:
        w.write(f"{json.dumps(outputs)}\n")
    logger.info(f"Output file {output_file}")


def plot_subspace_count_per_point():
    import json
    model = "knn"
    input_file = f"{model}_subspace_count_per_point.json"
    BIN_NUM = 10
    with open(input_file) as f:
        obj = json.loads(f.readlines()[0])
    for aggregate_threshold, data_rst in obj.items():
        aggregator, threshold = aggregate_threshold.split("_")
        file_name = f"{model}_subspace_count_per_point_{aggregator}_{threshold}.pdf"
        pp = PdfPages(file_name)
        print(aggregator, threshold)
        for dataset, data in data_rst.items():
            f, ax = plt.subplots(1, figsize=(4, 2))
            x_locs = np.arange(BIN_NUM + 1)
            width = 0.3

            outlier_hist_percent = data["outlier"]
            inlier_hist_percent = data["inlier"]

            bar1 = ax.bar(x_locs - width / 2, outlier_hist_percent, width, label="Outlier")
            bar2 = ax.bar(x_locs + width / 2, inlier_hist_percent, width, label="Inlier")
            ax.set_xticks(x_locs)
            ax.set_xticklabels(data["bin"])
            ax.legend()
            ax.xaxis.set_tick_params(labelsize=10)
            ax.set_ylabel("Percentage of Points")
            ax.set_xlabel("Ratio of Outlying Subspaces")
            autolabel(ax, bar1)
            autolabel(ax, bar2)
            plt.savefig(pp, format='pdf', bbox_inches="tight")
            print(
                f"{dataset} & {data['outlier_mean']:.0f} & {data['inlier_mean']:.0f} & {data['outlier_median']:.0f} & {data['inlier_median']:.0f} \\\\ \hline")
        pp.close()
        plt.close("all")
        logger.info(f"File name {file_name}")


def plot_point_count_per_dim():
    import json
    count_threshold = 2
    input_file = f"point_count_per_dim.json"
    with open(input_file) as f:
        obj = json.loads(f.readlines()[0])

    for aggregate_threshold, data_rst in obj.items():
        print(aggregate_threshold)
        aggregator, threshold = aggregate_threshold.split("_")
        file_name = f"point_count_per_dim_{aggregator}_{threshold}.pdf"
        pp = PdfPages(file_name)

        for dataset, data in data_rst.items():
            f, ax = plt.subplots(1, figsize=(4, 2))
            x_locs = np.arange(len(data["feature_index"]))
            width = 0.3
            bar1 = ax.bar(x_locs - width / 2, data["outlier"], width, label="Outlier")
            bar2 = ax.bar(x_locs + width / 2, data["inlier"], width, label="Inlier")
            ax.set_xticks(x_locs)
            ax.set_xticklabels([i + 1 for i in range(len(data["feature_index"]))])
            ax.xaxis.set_tick_params(labelsize=10)
            ax.legend(loc=[0.305, 1.01], ncol=2)
            logger.info(dataset)
            ax.set_ylabel("Percentage of Points")
            ax.set_xlabel("Dimension of Outlying Subspaces")
            autolabel(ax, bar1, 2)
            autolabel(ax, bar2, 2)
            plt.savefig(pp, format='pdf', bbox_inches="tight")

        pp.close()
        plt.close("all")
        logger.info(f"File name {file_name}")

def plot_outlier_per_subspace_hist():
    import json
    model = "knn"
    input_file = f"{model}_outliers_per_subspace.json"
    BIN_NUM = 10
    with open(input_file) as f:
        obj = json.loads(f.readlines()[0])

    for aggregate_threshold, data_rst in obj.items():
        aggregator, threshold = aggregate_threshold.split("_")
        file_name = f"{model}_outlier_count_per_subspace_{aggregator}_{threshold}.pdf"
        pp = PdfPages(file_name)

        for dataset, data in data_rst.items():
            outlier_hist, bin = np.histogram(np.array(data["outlier_dist"]), BIN_NUM, range=(0.1, data["outlier_total"]))
            zero_outlier_subspaces = sum([1 for i in data["outlier_dist"] if i == 0])
            outlier_hist = np.insert(outlier_hist, 0, zero_outlier_subspaces)
            outlier_hist = outlier_hist / data["subspace_total"]

            f, ax = plt.subplots(1, figsize=(4, 2))
            x_locs = np.arange(BIN_NUM + 1)

            width = 0.3
            bar = ax.bar(x_locs, outlier_hist, width)
            ax.set_xticks(x_locs)
            ax.set_xticklabels([i/10 for i in range(11)])
            ax.xaxis.set_tick_params(labelsize=10)
            # ax.legend(loc=[0.305, 1.01], ncol=2)
            logger.info(dataset)
            ax.set_ylabel("Subspaces Percentage")
            ax.set_xlabel("(Outliers in a subspace)/(Total Outliers)")
            ax.set_title()
            autolabel(ax, bar, 2)
            plt.savefig(pp, format='pdf', bbox_inches="tight")

        pp.close()
        plt.close("all")
        logger.info(f"File name {file_name}")

if __name__ == '__main__':
    plot_outlier_per_subspace_hist()
    # outliers_per_subspace()
