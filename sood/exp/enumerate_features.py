#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

from collections import defaultdict
from itertools import combinations
import time
import numpy as np
import matplotlib

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
    threshold = 0
    for dataset in [Dataset.VOWELS, Dataset.WINE,
                    Dataset.BREASTW, Dataset.VERTEBRAL, Dataset.ANNTHYROID,
                    Dataset.GLASS, Dataset.PIMA, Dataset.THYROID]:
        logger.info("=" * 50)
        logger.info(f"             Dataset {dataset}             ")
        logger.info("=" * 50)
        X, Y = DataLoader.load(dataset)

        model_outputs = []
        total_feature = X.shape[1]
        feature_index = np.array([i for i in range(total_feature)])

        neigh = max(10, int(np.floor(0.03 * X.shape[0])))
        mdl = kNN(neigh, Normalize.ZSCORE)
        # mdl = kNN_GPU(neigh, Normalize.ZSCORE)
        # mdl = GKE(Normalize.ZSCORE)

        for l in range(1, len(feature_index) + 1):
            for i in combinations(feature_index, l):
                selected_features = np.asarray(i)
                _X = X[:, selected_features]
                y_scores = mdl.fit(_X)
                local_roc = roc_auc_score(Y, y_scores)
                if local_roc > threshold:
                    model_outputs.append(y_scores)

        logger.info(f"Total model {len(model_outputs)}")
        if len(model_outputs) > 0:
            count_threshold = 0.05
            score = Aggregator.count_rank_threshold(model_outputs, count_threshold)
            y_scores = np.array(score)
            roc = roc_auc_score(Y, y_scores)
            precision = precision_n_scores(Y, y_scores)
            outlier_subspaces = y_scores[Y == 1]
            inlier_subspaces = y_scores[Y == 0]
            logger.info(f"Outliers: {outlier_subspaces.shape}")
            logger.info(f"Inliers: {inlier_subspaces.shape}")
            logger.info(
                f"Outlier Subspaces Min: {np.min(outlier_subspaces)} Max: {np.max(outlier_subspaces)} Mean: {np.mean(outlier_subspaces)}")
            logger.info(
                f"Inlier Subspaces Min: {np.min(inlier_subspaces)} Max: {np.max(inlier_subspaces)} Mean: {np.mean(inlier_subspaces)}")
            logger.info(f"ROC of Count top-{count_threshold} {roc} Precision {precision}")

            count_threshold = 2
            score = Aggregator.count_std_threshold(model_outputs, count_threshold)
            y_scores = np.array(score)
            roc = roc_auc_score(Y, y_scores)
            precision = precision_n_scores(Y, y_scores)
            logger.info(f"ROC of Count {count_threshold}-std {roc} Precision {precision}")

            score = Aggregator.average(model_outputs)
            y_scores = np.array(score)
            roc = roc_auc_score(Y, y_scores)
            precision = precision_n_scores(Y, y_scores)
            logger.info(f"ROC of Average {roc} Precision {precision}")

            average_threshold = 2
            score = Aggregator.average_threshold(model_outputs, average_threshold)
            y_scores = np.array(score)
            roc = roc_auc_score(Y, y_scores)
            precision = precision_n_scores(Y, y_scores)
            logger.info(f"ROC of Average {average_threshold}-std {roc} Precision {precision}")


def compare_dim_dist():
    import json
    outputs = {}
    count_threshold = 1
    for dataset in [Dataset.VOWELS, Dataset.WINE,
                    Dataset.BREASTW, Dataset.ANNTHYROID,
                    Dataset.GLASS, Dataset.PIMA, Dataset.THYROID]:
        logger.info("=" * 50)
        logger.info(f"             Dataset {dataset}             ")
        logger.info("=" * 50)
        _X, Y = DataLoader.load(dataset)
        feature_index = np.array([i for i in range(_X.shape[1])])
        X_gpu_tensor = GKE_GPU.convert_to_tensor(_X)

        outlier_num, inlier_num = np.sum(Y == 1), np.sum(Y == 0)

        mdl = GKE_GPU(Normalize.ZSCORE)

        subspace_dim_outlier_1, subspace_dim_inlier_1 = defaultdict(set), defaultdict(set)
        subspace_dim_outlier_2, subspace_dim_inlier_2 = defaultdict(set), defaultdict(set)

        outlier_dim_hist_1 = [0] * len(feature_index)
        inlier_dim_hist_1 = [0] * len(feature_index)
        outlier_dim_hist_2 = [0] * len(feature_index)
        inlier_dim_hist_2 = [0] * len(feature_index)

        for l in range(1, len(feature_index) + 1):
            model_outputs = []
            for i in combinations(feature_index, l):
                model_outputs.append(mdl.fit(X_gpu_tensor[:, np.asarray(i)]))

            for count_threshold in [1, 2]:
                y_scores = np.array(Aggregator.count_std_threshold(model_outputs, count_threshold))
                outlier_subspaces = y_scores[Y == 1]
                inlier_subspaces = y_scores[Y == 0]
                for idx, score in enumerate(outlier_subspaces):
                    if score > 0:
                        if count_threshold == 1:
                            subspace_dim_outlier_1[l].add(idx)
                        elif count_threshold == 2:
                            subspace_dim_outlier_2[l].add(idx)

                for idx, score in enumerate(inlier_subspaces):
                    if score > 0:
                        if count_threshold == 1:
                            subspace_dim_inlier_1[l].add(idx)
                        elif count_threshold == 2:
                            subspace_dim_inlier_2[l].add(idx)


        for l, points in subspace_dim_outlier_1.items():
            outlier_dim_hist_1[l - 1] = len(points) / outlier_num
        for l, points in subspace_dim_inlier_1.items():
            inlier_dim_hist_1[l - 1] = len(points) / inlier_num

        for l, points in subspace_dim_outlier_2.items():
            outlier_dim_hist_2[l - 1] = len(points) / outlier_num
        for l, points in subspace_dim_inlier_2.items():
            inlier_dim_hist_2[l - 1] = len(points) / inlier_num

        outputs[dataset] = {
            "outlier_1": outlier_dim_hist_1,
            "inlier_1": inlier_dim_hist_1,
            "outlier_2": outlier_dim_hist_2,
            "inlier_2": inlier_dim_hist_2,
            "feature_index": feature_index.tolist(),
        }

    with open(f"outlying_subspace_dist_std_{count_threshold}.json", "w") as w:
        w.write(f"{json.dumps(outputs)}")




def autolabel(ax, rects, digits=1):
    """Attach a text label above each bar in *rects*, displaying its height."""
    counter = 0
    for rect in rects:
        height = rect.get_height()
        if counter == 0:
            if height > 0.1:
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            counter += 1



def compare_hist_dist(count_threshold=None):
    import json
    threshold = 0
    BIN_NUM = 10
    outputs = {}
    if count_threshold is None:
        count_threshold = 2
    output_file = f"compare_hist_dist_std_{count_threshold}.json"

    for dataset in [Dataset.VOWELS, Dataset.WINE,
                    Dataset.BREASTW, Dataset.ANNTHYROID,
                    Dataset.GLASS, Dataset.PIMA, Dataset.THYROID]:
        logger.info("=" * 50)
        logger.info(f"             Dataset {dataset}             ")
        logger.info("=" * 50)
        _X, Y = DataLoader.load(dataset)
        feature_index = np.array([i for i in range(_X.shape[1])])
        # mdl = kNN(max(10, int(np.floor(0.03 * _X.shape[0]))), Normalize.ZSCORE)
        mdl = GKE_GPU(Normalize.ZSCORE)
        X_gpu_tensor = GKE_GPU.convert_to_tensor(_X)


        model_outputs = []
        for l in range(1, len(feature_index) + 1):
            for i in combinations(feature_index, l):
                score = mdl.fit(X_gpu_tensor[:, np.asarray(i)])
                if roc_auc_score(Y, np.array(Aggregator.average([score, ]))) > threshold:
                    model_outputs.append(score)

        logger.info(f"Total model {len(model_outputs)}")
        if len(model_outputs) > 0:
            y_scores = np.array(Aggregator.count_std_threshold(model_outputs, count_threshold))
            outlier_num, inlier_num = np.sum(Y == 1), np.sum(Y == 0)
            outlier_subspaces, inlier_subspaces = y_scores[Y == 1], y_scores[Y == 0]

            outlier_hist, bin = np.histogram(outlier_subspaces, BIN_NUM, range=(0, len(model_outputs)))
            bin = [f"{i / len(model_outputs):.1f}" for i in bin]
            inlier_hist = np.histogram(inlier_subspaces, BIN_NUM, range=(0, len(model_outputs)))[0]
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

            outputs[dataset] = {
                "outlier": outlier_hist_percent.tolist(),
                "inlier": inlier_hist_percent.tolist(),
                "bin": bin,
                "outlier_mean": np.mean(outlier_subspaces),
                "inlier_mean": np.mean(inlier_subspaces),
                "outlier_median": np.median(outlier_subspaces),
                "inlier_median": np.median(inlier_subspaces),
            }


    with open(output_file, "w") as w:
        w.write(f"{json.dumps(outputs)}\n")


def plot_compare_hist_dist():
    import json
    count_threshold = 2
    file_name = f"compare_hist_dist_std_{count_threshold}.pdf"
    pp = PdfPages(file_name)
    with open(f"compare_hist_dist_std_{count_threshold}.json") as f:
        ln = f.readlines()[0]
        obj = json.loads(ln)
    BIN_NUM = 10
    for dataset, data in obj.items():
        f, ax = plt.subplots(1, figsize=(4, 2))
        x_locs = np.arange(BIN_NUM)
        width = 0.5

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
        print(f"{dataset} & {data['outlier_mean']:.0f} & {data['inlier_mean']:.0f} & {data['outlier_median']:.0f} & {data['inlier_median']:.0f} \\\\ \hline")
    pp.close()
    logger.info(f"File name {file_name}")

def plot_dim_hist_dist():
    import json
    count_threshold = 2
    file_name = f"outlying_subspace_dist_std_{count_threshold}.pdf"
    pp = PdfPages(file_name)
    with open(f"outlying_subspace_dist_std_{count_threshold}.json") as f:
        ln = f.readlines()[0]
        obj = json.loads(ln)

    for dataset, data in obj.items():
        f, ax = plt.subplots(1)
        x_locs = np.arange(len(data["feature_index"]))
        width = 0.5
        bar1 = ax.bar(x_locs - width / 2, data["outlier"], width, label="Outlier")
        bar2 = ax.bar(x_locs + width / 2, data["inlier"], width, label="Inlier")
        ax.set_xticks(x_locs)
        ax.set_xticklabels([i + 1 for i in range(len(data["feature_index"]))])
        ax.legend(loc=[0.005, 1.01])
        ax.set_title(dataset)
        ax.set_ylabel("Percentage of Points")
        ax.set_xlabel("Dimension of Outlying Subspaces")
        autolabel(ax, bar1, 2)
        autolabel(ax, bar2, 2)
        plt.savefig(pp, format='pdf', bbox_inches="tight")

    pp.close()
    logger.info(f"File name {file_name}")

if __name__ == '__main__':
    # import torch
    # @torch.jit.script
    # def my_cdist(x1, x2):
    #     x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    #     x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    #     res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    #     res = res.clamp_min_(1e-30)
    #     print("1", res)
    #     res = res * -1.0
    #     print("2", res)
    #     res = torch.exp(res)
    #     print(res)
    #     print(res)
    #     return res
    # a = torch.tensor([[1,2], [3,4], [5,6]], dtype=torch.float64)
    # rst = my_cdist(a, a)
    # print(rst)
    # rst =  torch.cdist(a, a, 2)
    # print(rst)
    # rst = rst * rst * -1
    # print(rst)
    # rst = torch.exp(rst)
    # print(rst)
    # rst = torch.sum(rst, axis=1)
    # print(rst)
    compare_dim_dist()
