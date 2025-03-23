import numpy as np
from abc import ABCMeta, abstractmethod

class Regularizer(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, weights):
        raise NotImplementedError


class L1Regularizer(Regularizer):

    def __init__(self, l1=0.01):
        self.l1 = l1

    def __call__(self, weights):
        return self.l1 * np.sign(weights)