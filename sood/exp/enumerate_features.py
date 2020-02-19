#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

from itertools import combinations

import numpy as np
from pyod.utils import precision_n_scores
from sklearn.metrics import roc_auc_score

from sood.model.abs_model import Aggregator

from sood.data_process.data_loader import DataLoader, Dataset
from sood.model.base_detectors import kNN

X, Y = DataLoader.load(Dataset.THYROID)

model_outputs = []
total_feature = X.shape[1]
feature_index = np.array([i for i in range(total_feature)])

neigh = max(10, int(np.floor(0.03 * X.shape[0])))
mdl = kNN(neigh, None)

for l in range(1, len(feature_index) + 1):
    for i in combinations(feature_index, l):
        selected_features = np.asarray(i)
        _X = X[:, selected_features]
        model_outputs.append(mdl.fit(_X))
        print("Finish one")

print(f"Total model {len(model_outputs)}")


score = Aggregator.average_threshold(model_outputs, 2)
y_scores = np.array(score)
roc = roc_auc_score(Y, y_scores)
print(f"ROC of average 2-std {roc}")
precision = precision_n_scores(Y, y_scores)
print(f"Precision of average 2-std {precision}")


score = Aggregator.count_rank_threshold(model_outputs, int(X.shape[0] * 0.05) )
y_scores = np.array(score)
roc = roc_auc_score(Y, y_scores)
print(f"ROC of count top5% {roc}")
precision = precision_n_scores(Y, y_scores)
print(f"Precision of count top5% {precision}")

score = Aggregator.count_std_threshold(model_outputs, 2)
y_scores = np.array(score)
roc = roc_auc_score(Y, y_scores)
print(f"ROC of count std {roc}")
precision = precision_n_scores(Y, y_scores)
print(f"Precision of count std {precision}")


score = Aggregator.average(model_outputs)
y_scores = np.array(score)
roc = roc_auc_score(Y, y_scores)
print(f"ROC of average {roc}")
precision = precision_n_scores(Y, y_scores)
print(f"Precision of average {precision}")
