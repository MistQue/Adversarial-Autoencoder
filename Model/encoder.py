#! -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from Util.util import linear_layer, get_dim
from Util.batch_normalize import batch_norm

class Encoder(object):
    def __init__(self, layer_list, z_dim):
        self.layer_list = layer_list[0:-1]
        self.z_dim = layer_list[-1]        
        self.name_scope = u'encoder'
  
    def get_variables(self):
        t_var = tf.trainable_variables()
        ret = []
        for var in t_var:
            if self.name_scope in var.name:
                ret.append(var)
        return ret
    
    def set_model(self, figs, is_training):        
        h = data
        with tf.variable_scope(self.name_scope):
            for i, (in_dim, out_dim) in enumerate(zip(self.layer_list, self.layer_list[1:])):
                h = linear_layer(h, in_dim, out_dim, i)
                h = batch_norm(h, i, is_training)
                h = tf.nn.relu(h)

            mu = linear_layer(h, self.layer_list[-1], self.z_dim, 'mu')
            log_sigma = linear_layer(h, self.layer_list[-1], self.z_dim, 'log_sigma')
        return mu, log_sigma
    
if __name__ == u'__main__':
    e = Encoder([784, 1200, 600, 300, 100], 2)
    data = tf.placeholder(tf.float32, [None, 784])
    e.set_model(figs, True)
