import os
import sys
import numpy as np
import tensorflow as tf
from util import linear_layer, batch_norm

class Discriminator(object):
    def __init__(self, layer_list):
        self.layer_list = layer_list
        self.name_scope = 'discriminator'
        
    def get_variables(self):
        t_var = tf.trainable_variables()
        ret = []
        for var in t_var:
            if self.name_scope in var.name:
                ret.append(var)
        return ret
    
    def __call__(self, x, is_training, reuse):
        # return only logits
        h = x   
        with tf.variable_scope(self.name_scope, reuse=reuse):
            for i, (in_dim, out_dim) in enumerate(zip(self.layer_list, self.layer_list[1:-1])):
                h = linear_layer(h, in_dim, out_dim, i)
                h = batch_norm(h, i, is_training=is_training)
                h = tf.nn.relu(h)
                
            ret = linear_layer(h, self.layer_list[-2], self.layer_list[-1], 'output')            
        return ret
    
if __name__ == '__main__':
    disc = Discriminator([2, 50, 20, 10, 1])
    z = tf.placeholder(tf.float32, [None, 2])
    disc(z, True, False)
