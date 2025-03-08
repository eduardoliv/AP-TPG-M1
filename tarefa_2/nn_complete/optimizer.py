#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class Optimizer:

    def __init__(self, learning_rate = 0.01,  momentum = 0.90):
        self.retained_gradient = None
        self.learning_rate = learning_rate
        self.momentum = momentum
 
    def update(self, w, grad_loss_w):
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        #self.retained_gradient = grad_loss_w
        return w - self.learning_rate * self.retained_gradient

# Adapted by: Grupo 03

class SGD:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w, grad_loss_w):
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros_like(w)
        # momentum-based update
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        return w - self.lr * self.retained_gradient