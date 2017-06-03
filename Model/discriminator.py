import os
import sys
import numpy as np
import tensorflow as tf
from util import linear_layer, batch_norm, lrelu

class Discriminator(object):
    def __init__(self, layer_list):
        self.layer_list = layer_list
        self.name_scope = 'discriminator'
        
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name_scope)
    
    def __call__(self, x, is_training, reuse):
        # return only logits
        h = x   
        with tf.variable_scope(self.name_scope, reuse=reuse):
            for i, (in_dim, out_dim) in enumerate(zip(self.layer_list, self.layer_list[1:-1])):
                h = linear_layer(h, in_dim, out_dim, i)
                #h = batch_norm(h, i, is_training=is_training)
                h = lrelu(h)
                
            ret = linear_layer(h, self.layer_list[-2], self.layer_list[-1], 'output')            
        return ret
    
if __name__ == '__main__':
    disc = Discriminator([2, 50, 20, 10, 1])
    z = tf.placeholder(tf.float32, [None, 2])
    disc(z, True, False)
