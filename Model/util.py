import os
import sys
import tensorflow as tf
import numpy as np


def get_weights(name, shape, trainable=True):
    return tf.get_variable('weights{}'.format(name), shape,
                           initializer=tf.contrib.layers.xavier_initializer(),
                           trainable=trainable)

def get_biases(name, shape, value, trainable=True):
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
    weights = get_weights(l_id, [in_dim, out_dim])
    biases  = get_biases(l_id, [out_dim], .0)
    return tf.matmul(x, weights) + biases

def conv_layer(inputs, out_num, filter_width, filter_hight, stride, l_id):
    # ** NOTICE: weight shape is [hight, width, in_chanel, out_chanel] **
    weights = get_weights(l_id,
                          [filter_hight, filter_width, inputs.get_shape()[-1], out_num])
    
    biases = get_biases(l_id, [out_num], 0.0)
    
    conved = tf.nn.conv2d(inputs, weights,
                          strides=[1, stride,  stride,  1],
                          padding = 'SAME')
    
    return tf.nn.bias_add(conved, biases)


def deconv_layer(inputs, out_shape, filter_width, filter_hight, stride, l_id):
    
    # ** NOTICE: weight shape is [hight, width, out_chanel, in_chanel] **
    weights = get_weights(l_id,
                          [filter_hight, filter_width, out_shape[-1], inputs.get_shape()[-1]])
    
    biases = get_biases(l_id, [out_shape[-1]], 0.0)
    
    deconved = tf.nn.conv2d_transpose(inputs, weights, output_shape = out_shape,
                                      strides=[1, stride,  stride,  1])
    return tf.nn.bias_add(deconved, biases)

def batch_norm(x, name, is_training=True):
    decay_rate = 0.99
    
    shape = x.get_shape().as_list()
    dim = shape[-1]
    if len(shape) == 2:
        mean, var = tf.nn.moments(x, [0], name = 'moments_bn_{}'.format(name))
    elif len(shape) == 4:
        mean, var = tf.nn.moments(x, [0, 1, 2], name = 'moments_bn_{}'.format(name))

    avg_mean  = get_biases('avg_mean_bn_{}'.format(name), [1, dim], 0.0, False)
    avg_var = get_biases('avg_var_bn_{}'.format(name), [1, dim], 1.0, False)
    
    beta  = get_biases('beta_bn_{}'.format(name), [1, dim], 0.0)
    gamma = get_biases('gamma_bn_{}'.format(name), [1, dim], 1.0)

    if is_training:
        avg_mean_assign_op = tf.assign(avg_mean, decay_rate * avg_mean
                                       + (1 - decay_rate) * mean)
        avg_var_assign_op = tf.assign(avg_var,
                                      decay_rate * avg_var
                                      + (1 - decay_rate) * var)

        with tf.control_dependencies([avg_mean_assign_op, avg_var_assign_op]):
            ret = gamma * (x - mean) / tf.sqrt(1e-6 + var) + beta
    else:
        ret = gamma * (x - avg_mean) / tf.sqrt(1e-6 + avg_var) + beta
        
    return ret