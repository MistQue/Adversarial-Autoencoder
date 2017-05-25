import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Model.model import Model
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':

    # parameter
    batch_size = 200
    epoch_num = 200
    z_dim = 100

    save_path = os.getcwd() + '/Save'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # get_data
    print('-- get data--')
    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot = False) 
    input_dim = np.shape(mnist.train.images)[1]

    # make model
    print('-- make model --')
    model = Model(input_dim, z_dim)
    model.set_model()

    # training
    print('-- begin training --')
    data = mnist.train.images
    num_one_epoch = np.shape(mnist.train.images)[0] // batch_size

    record_rec = np.zeros(epoch_num)   
    record_gen = np.zeros(epoch_num)
    record_disc = np.zeros(epoch_num)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(epoch_num):

            print('** epoch {} begin **'.format(epoch))
            obj_rec, obj_gen, obj_disc = 0.0, 0.0, 0.0
            np.random.shuffle(data)
            for step in range(num_one_epoch):
                
                # get batch data
                #batch_data, _ = mnist.train.next_batch(batch_size)
                batch_data = data[step * batch_size: (step + 1) * batch_size]
                
                # train
                obj_rec += model.training_rec(sess, batch_data)
                
                batch_p_z = np.random.standard_normal(size = (batch_size, z_dim))
                obj_disc += model.training_disc(sess, batch_data, batch_p_z)
        
                obj_gen += model.training_gen(sess, batch_data)
                
                
                
                if step%100 == 0:
                    print('   step {}/{} end'.format(step, num_one_epoch));sys.stdout.flush()
                    
            record_rec[epoch] = obj_rec / float(num_one_epoch)
            record_gen[epoch] = obj_gen / float(num_one_epoch)
            record_disc[epoch] = obj_disc / float(num_one_epoch)        
            print('epoch:{}, rec_loss = {}, gen_loss = {}, disc_loss = {}'.format(epoch,
                                                                        record_rec[epoch],
                                                                        record_gen[epoch],
                                                                        record_disc[epoch]))
            saver.save(sess, './Save/model.ckpt')
    
    np.save('./Save/record_rec.npy', record_rec)
    np.save('./Save/record_gen.npy', record_gen)
    np.save('./Save/record_disc.npy', record_disc)
