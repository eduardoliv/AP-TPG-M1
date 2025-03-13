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