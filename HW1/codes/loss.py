from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        """Your codes here"""
        return 0.5 * np.mean(np.sum(np.square(input - target), axis=1), axis=0)

    def backward(self, input, target):
        """Your codes here"""
        batch_size = input.shape[0]
        return (input - target) / batch_size
