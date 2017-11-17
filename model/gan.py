# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 13:17:19 2017

@author: Jinjing
"""

import tensorflow as tf
import numpy as np
from matplotlib.pyplot import imsave

batch_size = 256

image_height = 28
image_width = 28
image_size = image_height * image_width

z_size = 100

h1_size = 150
h2_size = 300

def build_generator(z_prior):
    W_1 = tf.Variable(tf.truncated_normal([z_size,h1_size],stddev = 0.1), dtype = tf.float32)
    B_1 = tf.Variable(tf.zeros([h1_size]), dtype = tf.float32)
    H_1 = tf.nn.relu(tf.matmul(z_prior,W_1)+B_1)
    
    W_2 = tf.Variable(tf.truncated_normal([h1_size,h2_size],stddev = 0.1), dtype = tf.float32)
    B_2 = tf.Variable(tf.zeros([h2_size]), dtype = tf.float32)
    H_2 = tf.nn.relu(tf.matmul(H_1,W_2) + B_2)
    
    W_3 = tf.Variable(tf.truncated_normal([h2_size,image_size],stddev = 0.1), dtype = tf.float32)
    B_3 = tf.Variable(tf.zeros([image_size]), dtype = tf.float32)
    H_3 = tf.matmul(H_2,W_3) + B_3
    
    x_generator = tf.nn.sigmoid(H_3)
    g_params = [W_1,B_1,W_2,B_2,W_3,B_3]
    
    return x_generator,g_params
     
def build_discrimitor(x_real,x_generator,keep_prob):
    
    X = tf.concat([x_generator,x_real],0)
    
    W_1 = tf.Variable(tf.truncated_normal([image_size,h2_size],stddev = 0.1), dtype = tf.float32)
    B_1 = tf.Variable(tf.zeros([h2_size]), dtype = tf.float32)
    H_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X,W_1) + B_1), keep_prob)
    
    W_2 = tf.Variable(tf.truncated_normal([h2_size,h1_size],stddev = 0.1), dtype = tf.float32)
    B_2 = tf.Variable(tf.zeros([h1_size]), dtype = tf.float32)
    H_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(H_1,W_2) + B_2), keep_prob)
    
    W_3 = tf.Variable(tf.truncated_normal([h1_size,1],stddev = 0.1), dtype = tf.float32)
    B_3 = tf.Variable(tf.zeros([h1_size]), dtype = tf.float32)
    H_3 = tf.matmul(H_2,W_3) + B_3
    
    y_generator = tf.nn.sigmoid(tf.slice(H_3,[0,0],[batch_size,-1]))
    y_real  = tf.nn.sigmoid(tf.slice(H_3,[batch_size,0],[-1,-1]))
    d_params = [W_1,B_1,W_2,B_2,W_3,B_3]
    
    return y_real,y_generator,d_params
def show_image(x_generator,fname,grid_size = (8,8),grid_padding = 5):
    x_generator = x_generator.reshape(batch_size,image_height,image_width)
    grid_h = image_height * grid_size[0] + grid_padding * (grid_size[0] - 1)
    grid_w = image_height * grid_size[1] + grid_padding * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h,grid_w),dtype = np.uint8)
    for i,image in enumerate(x_generator):
        if i >= grid_size[0] * grid_size[1]:
            break
        image = image*255.
        image = image.astype(np.uint8)
        row = (i // grid_size[0]) * (image_height + grid_padding)
        col = (i % grid_size[1]) * (image_width + grid_padding)
        img_grid[row : row+image_height, col : col+image_width] = image
        imsave(fname,img_grid)
def train(to_restore):
    
    learn_rate = 0.0001
    max_epochs = 500
    x_sample = np.load('./phd08_data_1.npy')
    x_sample_size = x_sample.shape[0]
    x_sample = x_sample.reshape((x_sample_size,784))
    
    z_sample = np.random.normal(0,1,size = (batch_size,z_size)).astype(np.float32)
    
    x_real = tf.placeholder(tf.float32,[None,image_size])
    z_piror = tf.placeholder(tf.float32,[None,z_size])
    keep_prob = tf.placeholder(tf.float32)
    
    x_generator,g_params = build_generator(z_piror)
    
    y_real,y_generator,d_params = build_discrimitor(x_real,x_generator,keep_prob)
    
    g_loss = tf.reduce_mean(tf.log(y_real) + tf.log(1 - y_generator))# tf.reduce_mean(tf.log(y_real) + tf.log(1 - y_generator)) 和  tf.reduce_mean(-tf.log(y_generator))等价，因为更新g时，d不更新，tf.log(y_real)不变 
    d_loss = tf.reduce_mean(- (tf.log(y_real) + tf.log(1 - y_generator)))
     
    optimizer = tf.train.AdamOptimizer(learn_rate)  
    
    d_trainer = optimizer.minimize(d_loss,var_list = d_params)
    g_trainer = optimizer.minimize(g_loss,var_list = g_params)
    
    saver = tf.train.Saver()
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    if to_restore:
        chkpt_fname = tf.train.latest_checkpoint('./output')
        saver.restore(sess, chkpt_fname)

    for epoch in range(max_epochs):
        for i in range(int(x_sample_size // batch_size)):
            x_sample_each = x_sample[batch_size * i : batch_size * (i+1)]
            x_sample_each = x_sample_each / 255
            _,d_l = sess.run([d_trainer,d_loss],feed_dict = {x_real:x_sample_each,z_piror:z_sample,keep_prob:0.7})
            _,g_l = sess.run([g_trainer,d_loss],feed_dict = {x_real:x_sample_each,z_piror:z_sample,keep_prob:0.7})
        if epoch % 50 == 0:
            print('epoch: %s, d_loss: %s, g_loss: %s'%(epoch,d_l,g_l))
            x_generator_vaule = sess.run(x_generator,feed_dict = {z_piror : z_sample})
            show_image(x_generator_vaule,'./output/%s.jpg'% int(epoch//50))
            saver.save(sess,'./output/gda.model')
    sess.close()
def test(size):
    z_prior = tf.placeholder(tf.float32,[size,z_size])
    x_generator = build_generator(z_prior)
    chkpt_fname = tf.train.last_checkpoint('./ouput')
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess,chkpt_fname)
    
    z_test_value = np.random.normal(0, 1, size=(size, z_size)).astype(np.float32)
    x_gen_val = sess.run(x_generator, feed_dict={z_prior: z_test_value})
    show_image(x_gen_val, "./ouput/test_result.jpg")
if __name__ == '__main__':
    to_train = True
    if to_train:
        train(False)
    else:
        test(1)