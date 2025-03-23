#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Grupo 03
"""

from enum import Enum

class ModelRunMode(Enum):
    """
    Enumeration of Model Run Mode.
    """
    TRAIN           = "train"           # Train Mode
    CLASSIFY        = "classify"        # Classify Mode

class ModelType(Enum):
    """
    Enumeration of Model Types.
    """
    LOGREG = 1
    DNN    = 2
    RNN    = 3