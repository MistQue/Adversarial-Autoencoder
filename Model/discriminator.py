#! -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from Util.util import linear_layer
from Util.batch_normalize import batch_norm

class Discriminator(object):
    def __init__(self, layer_list):
        self.layer_list = layer_list
        self.name_scope = u'discriminator'
        
    def get_variables(self):
        t_var = tf.trainable_variables()
        ret = []
        for var in t_var:
            if self.name_scope in var.name:
                ret.append(var)
        return ret
    
    def set_model(self, z, is_training):
        # return only logits
        h = z   
        with tf.variable_scope(self.name_scope):
            for i, (in_dim, out_dim) in enumerate(zip(self.layer_list, self.layer_list[1:])):
                ret = linear_layer(h, in_dim, out_dim, i)
                h = batch_norm(ret, i, is_training)
                h = tf.nn.relu(h)            
        return ret
    
if __name__ == u'__main__':
    d = Discriminator([2, 50, 20, 10, 1])
    z = tf.placeholder(tf.float32, [None, 2])
    d.set_model(z, True)
