#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

from collections import defaultdict
from itertools import combinations

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
from sood.model.base_detectors import kNN, GKE, kNN_GPU
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
    pp = PdfPages("Outlying_Subspace_Dim_Dist.pdf")
    count_threshold = 0.05
    for dataset in [Dataset.VOWELS, Dataset.WINE,
                    Dataset.BREASTW, Dataset.VERTEBRAL, Dataset.ANNTHYROID,
                    Dataset.GLASS, Dataset.PIMA, Dataset.THYROID]:
        logger.info("=" * 50)
        logger.info(f"             Dataset {dataset}             ")
        logger.info("=" * 50)
        X, Y = DataLoader.load(dataset)

        feature_index = np.array([i for i in range(X.shape[1])])
        outlier_num, inlier_num = np.sum(Y == 1), np.sum(Y == 0)

        neigh = max(10, int(np.floor(0.03 * X.shape[0])))
        mdl = kNN(neigh, Normalize.ZSCORE)

        subspace_dim_outlier, subspace_dim_inlier = defaultdict(set), defaultdict(set)

        outlier_dim_hist = [0] * len(feature_index)
        inlier_dim_hist = [0] * len(feature_index)

        for l in range(1, len(feature_index) + 1):
            model_outputs = []
            for i in combinations(feature_index, l):
                model_outputs.append(mdl.fit(X[:, np.asarray(i)]))

            y_scores = np.array(Aggregator.count_rank_threshold(model_outputs, count_threshold))
            outlier_subspaces = y_scores[Y == 1]
            inlier_subspaces = y_scores[Y == 0]
            print(y_scores)
            for idx, score in enumerate(outlier_subspaces):
                if score > 0:
                    subspace_dim_outlier[l].add(idx)
            for idx, score in enumerate(inlier_subspaces):
                if score > 0:
                    subspace_dim_inlier[l].add(idx)

        for l, points in subspace_dim_outlier.items():
            outlier_dim_hist[l - 1] = len(points) / outlier_num
        for l, points in subspace_dim_inlier.items():
            inlier_dim_hist[l - 1] = len(points) / inlier_num

        f, ax = plt.subplots(1)
        x_locs = np.arange(len(feature_index))
        width = 0.5
        bar1 = ax.bar(x_locs - width / 2, outlier_dim_hist, width, label="Outlier")
        bar2 = ax.bar(x_locs + width / 2, inlier_dim_hist, width, label="Inlier")
        ax.set_xticks(x_locs)
        ax.set_xticklabels([i + 1 for i in range(len(feature_index))])
        ax.legend(loc=[0.005, 1.01])
        ax.set_title(dataset)
        ax.set_ylabel("Percentage of Points")
        ax.set_xlabel("Dimension of Outlying Subspaces")
        autolabel(ax, bar1, 2)
        autolabel(ax, bar2, 2)
        plt.savefig(pp, format='pdf', bbox_inches="tight")

        print(outlier_dim_hist)
        print(inlier_dim_hist)
    pp.close()


def autolabel(ax, rects, digits=1):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if height > 0.1:
            if height == 1:
                ax.annotate(f'{height:.0f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            else:
                if height >= 0.95:
                    ax.annotate(f'{height:.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
                else:
                    ax.annotate(f'{height:.1f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')


def compare_hist_dist():
    threshold = 0
    BIN_NUM = 10
    pp = PdfPages("Subspace_Count_Dist.pdf")

    for dataset in [Dataset.VOWELS, Dataset.WINE,
                    Dataset.BREASTW, Dataset.VERTEBRAL, Dataset.ANNTHYROID,
                    Dataset.GLASS, Dataset.PIMA, Dataset.THYROID]:
        logger.info("=" * 50)
        logger.info(f"             Dataset {dataset}             ")
        logger.info("=" * 50)
        X, Y = DataLoader.load(dataset)
        feature_index = np.array([i for i in range(X.shape[1])])
        neigh = max(10, int(np.floor(0.03 * X.shape[0])))
        mdl = kNN(neigh, Normalize.ZSCORE)
        count_threshold = 0.05

        model_outputs = []
        for l in range(1, len(feature_index) + 1):
            for i in combinations(feature_index, l):
                y_scores = np.array(Aggregator.count_rank_threshold([mdl.fit(X[:, np.asarray(i)]), ], count_threshold))
                if roc_auc_score(Y, y_scores) > threshold:
                    model_outputs.append(y_scores)

        logger.info(f"Total model {len(model_outputs)}")
        if len(model_outputs) > 0:
            y_scores = np.array(Aggregator.count_rank_threshold(model_outputs, count_threshold))
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

            f, ax = plt.subplots(1)
            x_locs = np.arange(BIN_NUM)
            width = 0.5
            bar1 = ax.bar(x_locs - width / 2, outlier_hist_percent, width, label="Outlier")
            bar2 = ax.bar(x_locs + width / 2, inlier_hist_percent, width, label="Inlier")
            ax.set_xticks(x_locs)
            ax.set_xticklabels(bin)
            ax.legend()
            ax.set_title(dataset)
            ax.set_ylabel("Percentage of Points")
            ax.set_xlabel("Ratio of Outlying Subspaces")
            autolabel(ax, bar1)
            autolabel(ax, bar2)
            plt.savefig(pp, format='pdf', bbox_inches="tight")
    pp.close()


if __name__ == '__main__':
    compare_auc()
