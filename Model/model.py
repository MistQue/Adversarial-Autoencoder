#! -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from encoder import Encoder
from decoder import Decoder
from discriminator import Discriminator


class Model(object):
    def __init__(self, input_dim, z_dim):
        
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.lr = 0.001
        
        # -- encoder -------
        self.encoder = Encoder([input_dim, 600, 300, 100, z_dim])
        
        # -- decoder -------
        self.decoder = Decoder([z_dim, 100, 300, 600, input_dim])

        # -- discriminator --
        self.discriminator = Discriminator([z_dim, 300, 300, 100, 1])
        
        
    def set_model(self):

        self.x = tf.placeholder(tf.float32, [None, self.input_dim])
        self.batch_size = tf.shape(self.x)[0]
        
        mu, log_sigma = self.encoder.set_model(self.x, is_training = True)

        eps = tf.random_normal([self.batch_size, self.z_dim])
        z = eps * tf.exp(log_sigma) + mu

        gen_data = self.decoder.set_model(z, is_training = True)
        
        
        reconstruct_error = tf.reduce_mean(
            tf.reduce_sum(tf.pow(gen_data - self.x, 2), [1]))
        
        aae_logits = self.discriminator.set_model(z, is_training = True)
        
        

        d_loss_from_gen = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = aae_logits,
                targets = tf.zeros_like(aae_logits)))
        
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = aae_logits,
                targets = tf.ones_like(aae_logits)))

    
        # == for sharing variables ===
        tf.get_variable_scope().reuse_variables()

        # discriminator
        self.z_input = tf.placeholder(dtype = tf.float32, shape = [None, self.z_dim])
        disc_logits = self.discriminator.set_model(self.z_input, is_training = True)
        
        d_loss_from_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = disc_logits,
                targets = tf.ones_like(disc_logits)))

        # -- train -----
        # update weights of autoencoder from reconstruct error
        self.obj_aae = reconstruct_error
        train_vars = self.encoder.get_variables()
        train_vars.extend(self.decoder.get_variables())
        self.train_aae  = tf.train.AdamOptimizer(self.lr).minimize(self.obj_aae, var_list = train_vars)

        # update weights of generator from discrimator 
        self.obj_gen = g_loss
        train_vars = self.encoder.get_variables()
        self.train_gen  = tf.train.AdamOptimizer(self.lr).minimize(self.obj_gen, var_list = train_vars)

        # update weights of discrimator 
        self.obj_disc = d_loss_from_gen + d_loss_from_real
        train_vars = self.discriminator.get_variables()
        self.train_disc  = tf.train.AdamOptimizer(self.lr).minimize(self.obj_disc, var_list = train_vars)
        
        # -- for using ---------------------
        self.mu, _  = self.encoder.set_model(self.x, is_training = False)
        self.generate_data = self.decoder.set_model(self.z_input, is_training = False)
        
    def training_aae(self, sess, data):
        _, obj_aae = sess.run([self.train_aae, self.obj_aae],
                                  feed_dict = {self.x: data})
        return obj_aae
        
    def training_gen(self, sess, data):
        _, obj_gen = sess.run([self.train_gen, self.obj_gen],
                                  feed_dict = {self.x: data})
        return obj_gen
    
    def training_disc(self, sess, data, p_z):
        _, obj_disc = sess.run([self.train_disc, self.obj_disc],
                                  feed_dict = {self.x: data,
                                               self.z_input:p_z})
        return obj_disc
    
    def encoding(self, sess, data):
        ret = sess.run(self.mu, feed_dict = {self.x: data})
        return ret
    
    def gen_data(self, sess, z):
        datas = sess.run(self.generate_data, feed_dict = {self.z_input: z})
        return datas
    
if __name__ == u'__main__':
    model = Model(28 * 28 * 1, 2)
    model.set_model()
    
