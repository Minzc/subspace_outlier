#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

class Consts:
    DATA = "data"
    LABEL = "label"

class PathManager:
    def __init__(self):
        self.dataset = "../dataset"
        self.output = "output"

    def get_output(self, dataset, sample_method, base_method):
        return f"{self.output}/{dataset}_{sample_method}_{base_method}.json"