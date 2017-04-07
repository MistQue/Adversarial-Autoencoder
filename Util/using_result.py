#! -*- coding:utf-8 -*-

import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import cv2
from Model import Model
from make_fig import get_batch
import math

class FigGenerator(object):
    def __init__(self, file_name, z_dim, batch_size):

        self.batch_size = batch_size
        
        self.model = Model(28 * 28 * 1, z_dim, 10, batch_size)
        self.model.set_model()
        saver = tf.train.Saver()
        self.sess = tf.Session()

        saver.restore(self.sess, file_name)
    def encoding(self, figs):
        return self.model.encoding(self.sess, figs)
    
    def __call__(self, z_inputs):
        assert(len(z_inputs) == self.batch_size)
        return self.model.figure_generate(self.sess, z_inputs)

if __name__ == u'__main__':

    # dump file
    dump_file = u'./model.dump'
    
    # parameter
    batch_size = 50
    z_dim = 2

    # figure generator
    fig_gen = FigGenerator(dump_file, z_dim, batch_size)
    
    #z = fig_gen.model.sampler([2] * batch_size)
    '''
    r = 2
    z = [[r * math.cos(2 * np.pi * i/20.0), r * math.sin(2 * np.pi * i / 20.0)] for i in range(batch_size)]
    figs = (fig_gen(z) + 1.0) * 127.5 
    print 'z' 
    print np.shape(figs)
    for i, fig in enumerate(figs):
        tmp = np.reshape(fig, (28, 28 , 1))
        cv2.imwrite(os.path.join('sample_result', '{}.jpg'.format(i)), tmp)
    '''
    
    with open('mnist_test.csv', 'r') as f:
        labels, figs = get_batch(f, batch_size)
    z = fig_gen.encoding(figs)
    indexes = np.argmax(np.asarray(labels), axis = 1)
    np.save('z.npy', z)
    np.save('label.npy', indexes)
    print 'ok'
    
    #plt.scatter(z[:, 0], z[:, 1], c = indexes)
    #plt.legend()
    #plt.savefig('test.png')
    #plt.show()
