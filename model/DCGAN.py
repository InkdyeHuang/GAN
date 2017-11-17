# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 13:17:19 2017

@author: Jinjing
"""
import tensorflow as tf
import numpy as np
from matplotlib.pyplot import imsave

image_height = 28
image_width = 28
image_size = image_height * image_width

z_size = 100
y_size = 10

batch_size = 256

to_restore = True

def leaky_relu(x,leak = 0.2):
    return tf.maximum(x,x*leak)


def build_generator(z_piror,y_condition,training = False):
    
    g_input = tf.concat([z_piror,y_condition],1) # g_input_size= [batch_size,z_size + y_size]
    
    input_channel = int(g_input.shape[1])
    W_full = tf.Variable(tf.truncated_normal([input_channel,128*4*4],stddev = 0.02), dtype = tf.float32)
    B_full = tf.Variable(tf.constant(0.0,shape=[128*4*4]), dtype = tf.float32)
    g_input = tf.nn.relu(tf.matmul(g_input,W_full) + B_full)
    g_input = tf.reshape(g_input,[batch_size,4,4,128])
    
    W_1 = tf.Variable(tf.truncated_normal([5,5,64,128],stddev = 0.02), dtype = tf.float32)
    B_1 = tf.Variable(tf.constant(0.0,shape=[64]), dtype = tf.float32)
    H_1 = tf.nn.relu(tf.nn.conv2d_transpose(g_input,W_1,[batch_size,7,7,64],strides = [1,2,2,1],padding='SAME') + B_1)
    
    W_2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev = 0.02), dtype = tf.float32)
    B_2 = tf.Variable(tf.constant(0.0,shape = [32]), dtype = tf.float32)
    H_2 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d_transpose(H_1,W_2,[batch_size,14,14,32],strides = [1,2,2,1],padding='SAME') + B_2,axis = 0,training = training))
    
    W_3 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev = 0.02), dtype = tf.float32)
    B_3 = tf.Variable(tf.constant(0.0,shape=[1]), dtype = tf.float32)
    H_3 = tf.nn.conv2d_transpose(H_2,W_3,[batch_size,28,28,1],strides = [1,2,2,1],padding = 'SAME') + B_3
    
    x_generator = tf.nn.tanh(H_3)#激活函数？
    
    g_params = [W_1,B_1,W_2,B_2,W_3,B_3]
    
    return x_generator,g_params

def build_discrimitor(x_real,x_generator,y_condition,training = False):
    
    X = tf.concat([x_real,x_generator],0)
    
    W_1 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev = 0.02), dtype = tf.float32)
    B_1 = tf.Variable(tf.constant(0.0,shape=[32]), dtype = tf.float32)
    H_1 = leaky_relu(tf.nn.conv2d(X,W_1,strides=[1,2,2,1],padding='SAME') + B_1)
    
    W_2= tf.Variable(tf.truncated_normal([5,5,32,64],stddev = 0.02), dtype = tf.float32)
    B_2 = tf.Variable(tf.constant(0.0,shape=[64]), dtype = tf.float32)
    H_2 = leaky_relu(tf.nn.conv2d(H_1,W_2,strides=[1,2,2,1],padding='SAME') + B_2)
    
    W_3 = tf.Variable(tf.truncated_normal([5,5,64,128],stddev = 0.02), dtype = tf.float32)
    B_3 = tf.Variable(tf.constant(0.0,shape=[128]), dtype = tf.float32)
    H_3 = leaky_relu(tf.nn.conv2d(H_2,W_3,strides=[1,2,2,1],padding='SAME') + B_3)
    
    H_3 = tf.nn.avg_pool(H_3,[1,4,4,1],[1,1,1,1],padding = 'VALID')
    
    H_3_flat = tf.reshape(H_3,[batch_size * 2,-1])
    output_real = tf.concat([tf.slice(H_3_flat,[0,0],[batch_size,-1]),y_condition],1)
    output_generator = tf.concat([tf.slice(H_3_flat,[batch_size,0],[-1,-1]),y_condition],1)
    
    W_output = tf.Variable(tf.truncated_normal([y_size + 128,1],stddev = 0.02), dtype = tf.float32)
    B_output = tf.Variable(tf.constant(0.0,shape = [1]),dtype = tf.float32)
    y_real = tf.nn.sigmoid(tf.matmul(output_real,W_output) + B_output)
    y_generator = tf.nn.sigmoid(tf.matmul(output_generator,W_output) + B_output)
    
    d_params = [W_1,B_1,W_2,B_2,W_3,B_3,W_output,B_output]
    
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

def one_hot(lable,size):
    length = len(lable)
    lables = np.zeros([length,size],dtype = np.float32)
    for idx,i in enumerate(lable):
        lables[idx][i] = 1.0
    return lables
def train(to_restore):
    
    learn_rate = 0.0001
    max_epochs = 500
    x_sample = np.load('./phd08_data_1.npy')
    y_sample = np.load('./phd08_labels_1.npy')
    y_sample = one_hot(y_sample,y_size)
    x_sample_size = x_sample.shape[0]
    x_sample = x_sample.reshape([x_sample_size,28,28,1])
    
    z_sample = np.random.normal(0,1,size = (batch_size,z_size)).astype(np.float32)
    #y_sample_1 = np.random.normal(0,1,size = (batch_size,1,1,y_size)).astype(np.float32)
    #y_sample_2 = y_sample_1.reshape(-1,y_size)
    
    x_real = tf.placeholder(tf.float32,[None,28,28,1])
    z_piror = tf.placeholder(tf.float32,[None,z_size])
    y_condition = tf.placeholder(tf.float32,[None,y_size])
    
    x_generator,g_params = build_generator(z_piror,y_condition,training = True)
    
    y_real,y_generator,d_params = build_discrimitor(x_real,x_generator,y_condition,training = True)
    
    g_loss = tf.reduce_mean(-tf.log(y_generator))
    d_loss = tf.reduce_mean(- (tf.log(y_real) + tf.log(1 - y_generator)))
     
    optimizer = tf.train.GradientDescentOptimizer(learn_rate)  
    
    d_trainer = optimizer.minimize(d_loss,var_list = d_params)
    g_trainer = optimizer.minimize(g_loss,var_list = g_params)
    saver = tf.train.Saver(tf.global_variables())
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    if to_restore:
        chkpt_fname = tf.train.latest_checkpoint('./output')
        saver.restore(sess, chkpt_fname)
        
    y_sample_each = []
    for epoch in range(max_epochs):
        for i in range(int(x_sample_size // batch_size)):
            x_sample_each = x_sample[batch_size * i : batch_size * (i+1)]
            y_sample_each = y_sample[batch_size * i : batch_size * (i+1)]/255.
            x_sample_each = x_sample_each/255.
            _,d_l = sess.run([d_trainer,d_loss],feed_dict = {x_real:x_sample_each,z_piror:z_sample,y_condition:y_sample_each})
            _,g_l = sess.run([g_trainer,g_loss],feed_dict = {x_real:x_sample_each,z_piror:z_sample,y_condition:y_sample_each})
            print('epoch: %s,i:%s, d_loss: %s, g_loss: %s'%(epoch,i,d_l,g_l))
        if epoch % 5 == 0:
            x_generator_vaule = sess.run(x_generator,feed_dict = {z_piror : z_sample,y_condition:y_sample_each})
            show_image(x_generator_vaule,'./output/dcgan_%s.jpg'% int(epoch//5))
            saver.save(sess,'./output/dcgan.model')
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
    y_test = one_hot([1])
    x_gen_val = sess.run(x_generator, feed_dict={z_prior: z_test_value, y_condition :y_test})
    show_image(x_gen_val, "./ouput/test_result.jpg")
if __name__ == '__main__':
    to_train = True
    if to_train:
        train(False)
    else:
        test(1)