#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

import tqdm
from sood.data_process.data_loader import DataLoader, Dataset
from sood.log import getLogger
from sood.model.abs_model import AbstractModel, Aggregator
from sood.util import Similarity, PathManager, Consts
import numpy as np

from sood.model.base_detectors import kNN

logger = getLogger(__name__)


class OracleAdaptive(AbstractModel):
    NAME = "OracleAdaptive"

    def __init__(self, dim_start, dim_end, ensemble_size, aggregate_method, neighbor, base_model, Y, threshold):
        name = f"{self.NAME}({dim_start}-{dim_end} Neighbor: {neighbor}))"
        super().__init__(name, aggregate_method, base_model, neighbor)
        self.dim_start = dim_start
        self.dim_end = dim_end
        self.ensemble_size = ensemble_size
        self.aggregate_method = aggregate_method
        self.Y = Y
        self.threshold = threshold
        np.random.seed(1)

    def compute_ensemble_components(self, data_array):
        model_outputs = []
        total_feature = data_array.shape[1]
        feature_index = np.array([i for i in range(total_feature)])
        initial_count = int(self.ensemble_size * 0.05)
        selected_features = None
        local_rank_list = None
        rocs = []

        for i in range(initial_count):
            feature_size = np.random.randint(self.dim_start, self.dim_end)  # Randomly sample feature size
            selected_features = np.random.choice(feature_index, feature_size)  # Randomly select features
            local_rank_list = self.mdl.fit(data_array[:, selected_features])
            model_outputs.append(local_rank_list)
            logger.debug(f"Outlier score shape: {local_rank_list.shape}")

            local_roc = self.compute_roc_auc(np.array(self.aggregate_components([local_rank_list, ])), self.Y)
            print(f"Local {local_roc}")
            if local_roc > self.threshold:
                rocs.append(local_roc)

        global_rank_list = np.array(self.aggregate_components(model_outputs))

        for i in range(initial_count, self.ensemble_size):
            logger.info(f"Select features {selected_features}")
            s_score = Similarity.spearman(global_rank_list, local_rank_list)  # Compute spearman correlation

            choice_probability = [1] * total_feature
            for f_idx in selected_features:
                choice_probability[f_idx] += s_score
            choice_probability = np.exp(choice_probability)
            normalizer = sum(choice_probability)
            choice_probability = choice_probability / normalizer
            logger.info(f"Correlation {s_score}")

            feature_size = np.random.randint(self.dim_start, self.dim_end)  # Randomly sample feature size
            selected_features = np.random.choice(feature_index, feature_size,
                                                 p=choice_probability, replace=False)  # Randomly select features
            local_rank_list = self.mdl.fit(data_array[:, selected_features])

            global_rank_list = np.array(self.aggregate_components(model_outputs))
            roc_auc = self.compute_roc_auc(global_rank_list, self.Y)
            logger.info("Ensemble Before {}".format(roc_auc))

            local_roc = self.compute_roc_auc(np.array(self.aggregate_components([local_rank_list, ])), self.Y)
            logger.info(f"Local {local_roc}")
            if local_roc > self.threshold:
                rocs.append(local_roc)
                model_outputs.append(local_rank_list)
                global_rank_list = np.array(self.aggregate_components(model_outputs))
                roc_auc = self.compute_roc_auc(global_rank_list, self.Y)
                logger.info("Ensemble After {}".format(roc_auc))
            logger.info('-' * 50)
        logger.info("Number of good subspace {}/{}".format(len(rocs), self.ensemble_size))
        logger.info("Maximum roc {}".format(max(rocs)))
        logger.info("Minimum roc {}".format(min(rocs)))
        return model_outputs

    def aggregate_components(self, model_outputs):
        if self.aggregate_method == Aggregator.COUNT_RANK_THRESHOLD:
            return Aggregator.count_rank_threshold(model_outputs, 0.05)
        elif self.aggregate_method == Aggregator.AVERAGE:
            return Aggregator.average(model_outputs)
        elif self.aggregate_method == Aggregator.COUNT_STD_THRESHOLD:
            return Aggregator.count_std_threshold(model_outputs, 2)
        elif self.aggregate_method == Aggregator.AVERAGE_THRESHOLD:
            return Aggregator.average_threshold(model_outputs, 2)


def single_test():
    dataset = Dataset.BANK
    aggregator = Aggregator.AVERAGE
    threshold = 0

    X, Y = DataLoader.load(dataset)
    dim = X.shape[1]
    neigh = max(10, int(np.floor(0.03 * X.shape[0])))
    ENSEMBLE_SIZE = 100
    logger.info(f"{dataset} {aggregator} {threshold}")
    roc_aucs = []
    precision_at_ns = []
    mdl = OracleAdaptive(2, dim / 4, ENSEMBLE_SIZE, aggregator, neigh, kNN.NAME, Y, threshold)
    for _ in tqdm.trange(1):
        try:
            rst = mdl.run(X)
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


def batch_test():
    import json
    path_manager = PathManager()
    ENSEMBLE_SIZE = 100
    for dataset in [Dataset.ARRHYTHMIA, Dataset.MUSK, Dataset.MNIST_ODDS, Dataset.OPTDIGITS]:
        for aggregator in [Aggregator.AVERAGE, Aggregator.AVERAGE_THRESHOLD, Aggregator.COUNT_STD_THRESHOLD,
                           Aggregator.COUNT_RANK_THRESHOLD]:
            for base_model in [kNN.NAME, ]:
                # =======================================================================================
                # Model
                output_path = path_manager.get_batch_test_model_output(OracleAdaptive.NAME, aggregator, base_model,
                                                                       "DEFAULT", dataset)
                # =======================================================================================
                with open(output_path, "w") as w:
                    for threshold in [0, ]:
                        X, Y = DataLoader.load(dataset)
                        dim = X.shape[1]
                        neigh = max(10, int(np.floor(0.03 * X.shape[0])))
                        logger.info(f"{dataset} {aggregator} {threshold}")
                        roc_aucs = []
                        precision_at_ns = []
                        # =======================================================================================
                        # Model
                        mdl = OracleAdaptive(2, dim / 4, ENSEMBLE_SIZE, aggregator, neigh, base_model, Y,
                                             threshold)
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
                            Consts.START_DIM: 2,
                            Consts.END_DIM: 1 / 4,
                            Consts.ENSEMBLE_SIZE: ENSEMBLE_SIZE
                        }
                        w.write(f"{json.dumps(output)}\n")
                        logger.info(f"Output file is {output_path}")


if __name__ == '__main__':
    single_test()
