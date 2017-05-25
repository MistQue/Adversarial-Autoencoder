import os
import sys
import numpy as np
import tensorflow as tf
from util import linear_layer, batch_norm

class Decoder(object):
    def __init__(self, layer_list):
        self.layer_list = layer_list
        self.name_scope = 'decoder'

    def get_variables(self):
        t_var = tf.trainable_variables()
        ret = []
        for var in t_var:
            if self.name_scope in var.name:
                ret.append(var)
        return ret
    
    def __call__(self, x, is_training, reuse):
        h = x
        with tf.variable_scope(self.name_scope, reuse=reuse):
            for i, (in_dim, out_dim) in enumerate(zip(self.layer_list, self.layer_list[1:-1])):
                h = linear_layer(h, in_dim, out_dim, i)
                h = batch_norm(h, i, is_training=is_training)
                h = tf.nn.relu(h)
                
            h = linear_layer(h, self.layer_list[-2], self.layer_list[-1], 'output')
            h = batch_norm(h, 'output', is_training=is_training)
            ret = tf.nn.sigmoid(h)
        return ret
    
if __name__ == '__main__':
    dec = Decoder([2, 100, 600, 1200, 784])
    z = tf.placeholder(tf.float32, [None, 2])
    dec(z, True, False)
