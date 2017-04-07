#! -*- coding:utf-8 -*-

import os
import sys

import tensorflow as tf
import numpy as np
from math import *


def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

def get_weights(name, shape, stddev, trainable = True):
    return tf.get_variable('weights{}'.format(name), shape,
                           initializer = tf.random_normal_initializer(stddev = stddev),
                           trainable = trainable)

def get_biases(name, shape, value, trainable = True):
    return tf.get_variable('biases{}'.format(name), shape,
                           initializer = tf.constant_initializer(value),
                           trainable = trainable)
						   
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)
  
def get_dim(target):
    dim = 1
    for d in target.get_shape()[1:].as_list():
        dim *= d
    return dim

def linear_layer(x, in_dim, out_dim, l_id):
    weights = get_weights(l_id, [in_dim, out_dim], 1.0/np.sqrt(float(in_dim)))
    biases  = get_biases(l_id, [out_dim], 0.0)
    return tf.matmul(x, weights) + biases

def conv_layer(inputs, out_num, filter_width, filter_hight, stride, l_id):
    # ** NOTICE: weight shape is [hight, width, in_chanel, out_chanel] **
    weights = get_weights(l_id,
                          [filter_hight, filter_width, inputs.get_shape()[-1], out_num],
                          0.02)
    
    biases = get_biases(l_id, [out_num], 0.0)
    
    conved = tf.nn.conv2d(inputs, weights,
                          strides=[1, stride,  stride,  1],
                          padding = 'SAME')
    
    return tf.nn.bias_add(conved, biases)


def deconv_layer(inputs, out_shape, filter_width, filter_hight, stride, l_id):
    # ** NOTICE: weight shape is [hight, width, out_chanel, in_chanel] **
    weights = get_weights(l_id,
                          [filter_hight, filter_width, out_shape[-1], inputs.get_shape()[-1]],
                          0.02)
    
    biases = get_biases(l_id, [out_shape[-1]], 0.0)
    
    deconved = tf.nn.conv2d_transpose(inputs, weights, output_shape = out_shape,
                                      strides=[1, stride,  stride,  1])
    return tf.nn.bias_add(deconved, biases)

def sampler(batchsize, z_dim, batch_indices, n_class):
    if z_dim % 2 != 0:
        raise Exception("z_dim must be a multiple of 2.")

    def sample(x, y, label, n_class):
        shift = 1.4
        r = 2.0 * np.pi / float(n_class) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        x = np.array([new_x, new_y])
        return x.reshape((2,))

    x_var = 0.5
    y_var = 0.05
    x = np.random.normal(0, x_var, (batchsize, z_dim / 2))
    y = np.random.normal(0, y_var, (batchsize, z_dim / 2))
    z = np.empty((batchsize, z_dim), dtype=np.float32)
    
    for batch in xrange(batchsize):
        for zi in xrange(z_dim / 2):
            z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], batch_indices[batch], n_class)
    return z
