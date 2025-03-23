#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: miguelrocha
(Adapted by: Grupo 03)
"""

import numpy as np

class Math:
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
