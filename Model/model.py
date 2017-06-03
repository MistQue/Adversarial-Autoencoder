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
        self.enc_layer_list = [input_dim, 1000, 1000, z_dim]
        self.dec_layer_list = [z_dim, 1000, 1000, input_dim]
        self.disc_layer_list = [z_dim, 500, 500, 1]
        self.rec_lr = 5e-5
        self.gen_lr = 2e-5
        self.disc_lr = 2e-5
        self.mode = 'deterministic'
        
        # -- encoder -------
        self.encoder = Encoder(self.enc_layer_list)
        
        # -- decoder -------
        self.decoder = Decoder(self.dec_layer_list)

        # -- discriminator --
        self.discriminator = Discriminator(self.disc_layer_list)
        
    def set_model(self):

        self.x = tf.placeholder(tf.float32, [None, self.input_dim])
        self.p_z = tf.placeholder(dtype = tf.float32, shape = [None, self.z_dim])
        self.batch_size = tf.shape(self.x)[0]
        
        # ----- encoding -----
        mu, log_sigma = self.encoder(self.x, is_training=True, reuse=False)
        if self.mode == 'Non-deterministic':
            eps = tf.random_normal([self.batch_size, self.z_dim])
            q_z = eps * tf.exp(log_sigma) + mu
        elif self.mode == 'deterministic':
            q_z = mu
        
        
        # ----- decoding -----
        rec_x = self.decoder(q_z, is_training=True, reuse=False)
               
            
        # ----- loss -----
        reconstruct_error = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(rec_x - self.x), 1))

        real_logits = self.discriminator(self.p_z, is_training=True, reuse=False)
        fake_logits = self.discriminator(q_z, is_training=True, reuse=True)
        
        d_loss_from_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=real_logits, 
                labels=tf.ones_like(real_logits)))
        
        d_loss_from_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=fake_logits,
                labels = tf.zeros_like(fake_logits)))
        
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=fake_logits,
                labels=tf.ones_like(fake_logits)))

        
        # ----- train -----
        self.obj_rec = reconstruct_error
        train_vars = self.encoder.get_variables()
        train_vars.extend(self.decoder.get_variables())
        self.train_rec  = tf.train.RMSPropOptimizer(self.rec_lr, decay=0.5).minimize(self.obj_rec, var_list=train_vars)

        self.obj_gen = g_loss
        train_vars = self.encoder.get_variables()
        self.train_gen  = tf.train.RMSPropOptimizer(self.gen_lr, decay=0.5).minimize(self.obj_gen, var_list=train_vars)


        self.obj_disc =  d_loss_from_real + d_loss_from_fake
        train_vars = self.discriminator.get_variables()
        self.train_disc  = tf.train.RMSPropOptimizer(self.disc_lr, decay=0.5).minimize(self.obj_disc, var_list=train_vars)
        
        
        # ---- for using ----  
        self.gen_z, _  = self.encoder(self.x, is_training=False, reuse=True)
        self.gen_x = self.decoder(self.p_z, is_training=False, reuse=True)
        
    def training_rec(self, sess, data):
        _, obj_rec = sess.run([self.train_rec, self.obj_rec], 
                              feed_dict={self.x: data})
        return obj_rec
        
    def training_gen(self, sess, data):
        _, obj_gen = sess.run([self.train_gen, self.obj_gen], 
                              feed_dict={self.x: data})
        return obj_gen
    
    def training_disc(self, sess, data, p_z):
        _, obj_disc = sess.run([self.train_disc, self.obj_disc], 
                               feed_dict={self.x: data, self.p_z: p_z})
        return obj_disc
    
    def encoding(self, sess, data):
        ret = sess.run(self.gen_z, feed_dict={self.x: data})
        return ret
    
    def decoding(self, sess, z):
        ret = sess.run(self.gen_x, feed_dict={self.p_z: z})
        return ret

    def setting(self):      
        setting = {'input_dim': self.input_dim,
                   'z_dim': self.z_dim,
                   'enc_layer_list': self.enc_layer_list, 
                   'dec_layer_list': self.dec_layer_list,
                   'disc_layer_list': self.disc_layer_list,
                   'rec_lr': self.rec_lr,
                   'gen_lr': self.gen_lr,
                   'disc_lr': self.disc_lr,
                   'mode': self.mode}
        
        return setting
    
if __name__ == '__main__':
    model = Model(784, 2)
    model.set_model()
    
