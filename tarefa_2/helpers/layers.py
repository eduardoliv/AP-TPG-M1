#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: miguelrocha
(Adapted by: Grupo 03)
"""

from abc import ABCMeta, abstractmethod
import numpy as np
import copy

class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self, input):
        raise NotImplementedError
    
    @abstractmethod
    def backward_propagation(self, error):
        raise NotImplementedError
    
    @abstractmethod
    def output_shape(self):
        raise NotImplementedError
    
    @abstractmethod
    def parameters(self):
        raise NotImplementedError
    
    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape
    
    def layer_name(self):
        return self.__class__.__name__
    

class DenseLayer(Layer):
    
    def __init__(self, n_units, input_shape=None, dropout_rate=0.0, kernel_regularizer=None, bias_regularizer=None):
        super().__init__()
        self.n_units = n_units
        self._input_shape = input_shape
        self.dropout_rate = dropout_rate  # Add dropout rate
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        
        self.input = None
        self.output = None
        self.weights = None
        self.biases = None
        self.dropout_mask = None  # Store the dropout mask

    def initialize(self, optimizer):
        self.weights = np.random.rand(self.input_shape()[0], self.n_units) - 0.5
        self.biases = np.zeros((1, self.n_units))
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self
    
    def parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, inputs, training=True):
        self.input = inputs
        self.output = np.dot(self.input, self.weights) + self.biases
        
        if training and self.dropout_rate > 0:
            self.dropout_mask = (np.random.rand(*self.output.shape) >= self.dropout_rate).astype(np.float32)
            self.output *= self.dropout_mask  # Apply dropout by zeroing out some neurons
        elif not training and self.dropout_rate > 0:
            self.output *= (1.0 - self.dropout_rate)  # Scale outputs during inference
            
        return self.output
 
    def backward_propagation(self, output_error):
        if self.dropout_rate > 0:
            output_error *= self.dropout_mask  # Apply the same dropout mask during backpropagation
    
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        if self.kernel_regularizer:
            weights_error += self.kernel_regularizer(self.weights)
        if self.bias_regularizer:
            bias_error += self.bias_regularizer(self.biases)

        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)
        return input_error
 
    def output_shape(self):
        return (self.n_units,) 
