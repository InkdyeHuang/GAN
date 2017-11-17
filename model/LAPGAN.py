# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 13:17:19 2017

@author: Jinjing
"""
import tensorflow as tf
import numpy as np
from matplotlib.pyplot import imsave
import cv2

image_height = 28
image_width = 28
image_size = image_height * image_width

batch_size = 256

max_epochs = 500
n_level = 3
learn_rate = 0.0002
z_size = 100
y_size = 10
condition=True

to_restore = True

class con2v_layer():
    def __init__(self,filters,kerneal_size,strides):
        self.filters = filters
        self.kerneal_size = kerneal_size
        self.strides = strides

class Generator():
    def __init__(self,g_layers):
        self.name = 'Generator'
        self.g_layers = g_layers
        #self.k_initializer = tf.truncated_normal_initializer(stddev=0.01)
        #self.b_initializer = tf.zeros_initializer()
    def call(self,z_piror,y_condition,condition=True):
        if condition:
            g_input = tf.concat([z_piror,y_condition],3)
        else:
            g_input = z_piror
        #with tf.variable_scope(self.name) as scope:
        #print('lll')
        H_1 = tf.layers.conv2d_transpose(g_input,self.g_layers[0].filters,self.g_layers[0].kerneal_size,
                                         strides = self.g_layers[0].strides,padding = 'VALID',activation=tf.nn.relu,
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),bias_initializer=tf.zeros_initializer())
        H_2 = tf.layers.conv2d_transpose(H_1,self.g_layers[1].filters,self.g_layers[1].kerneal_size,
                                         strides = self.g_layers[1].strides,padding = 'SAME',activation=tf.nn.relu,
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),bias_initializer=tf.zeros_initializer())
        H_3 = tf.layers.conv2d_transpose(H_2,self.g_layers[2].filters,self.g_layers[2].kerneal_size,
                                         strides = self.g_layers[2].strides,padding = 'SAME',activation=tf.nn.sigmoid,
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),bias_initializer=tf.zeros_initializer())
        return H_3
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = self.name)
    
class Discrimitor():
    def __init__(self,d_layers):
        self.name = 'Discrimitor'
        self.d_layers = d_layers
    def call(self,x,y_condition,condition=True):
        #with tf.variable_scope(self.name) as scope:
        print('lll',self.d_layers[0].filters,self.d_layers[0].kerneal_size,self.d_layers[0].strides,tf.shape(x))
        H_1 = tf.layers.conv2d(x,self.d_layers[0].filters,self.d_layers[0].kerneal_size,
                               strides = self.d_layers[0].strides,padding = 'SAME',activation=tf.nn.relu,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),bias_initializer=tf.zeros_initializer())
        H_2 = tf.layers.conv2d(H_1,self.d_layers[1].filters,self.d_layers[1].kerneal_size,
                               strides = self.d_layers[1].strides,padding = 'SAME',activation=tf.nn.relu,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),bias_initializer=tf.zeros_initializer())
        H_3 = tf.layers.conv2d(H_2,self.d_layers[2].filters,self.d_layers[2].kerneal_size,
                               strides = self.d_layers[2].strides,padding = 'VALID',activation=tf.nn.sigmoid,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),bias_initializer=tf.zeros_initializer())
        H_3_flat = tf.reshape(H_3,[-1,self.d_layers[2].filters])
        if condition:
            y = tf.concat([H_3_flat,y_condition],1)
        else:
            y = H_3_flat
        W_output = tf.Variable(tf.truncated_normal([self.d_layers[2].filters*2,1],stddev = 0.2), dtype = tf.float32)
        B_output = tf.Variable(tf.constant(0.0,shape = [1]),dtype = tf.float32)
        y = tf.nn.sigmoid(tf.matmul(y,W_output) + B_output)
        return y
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = self.name)
class LAPGAN():
    def __init__(self,n_level,g_layers,d_layers):
        self.n_level = n_level
        self.g_layers = g_layers
        self.d_layer = d_layers
        self.Dis_models = []
        self.Gen_models = []
        
        for level in range(self.n_level):
            self.Dis_models.append(Discrimitor(d_layers[level]))
            self.Gen_models.append(Generator(g_layers[level]))
            
    def generate(self,sess,batchsize,z_sample,y_sample_1,condition=True,generator=False):
        print('hhh')
        z_piror = tf.placeholder(tf.float32,[None,1,1,z_size])
        y_condition  = [tf.placeholder(tf.float32,[None,1,1,y_size]),tf.placeholder(tf.float32,[None,1,1,14*14]),tf.placeholder(tf.float32,[None,1,1,28*28])]
        self.generator_outputs = []#i
        self.condition_set = []
        self.outputs = []#h
        for level in range(self.n_level):
            Gen_model = self.Gen_models[level]
            
            if level == 0:
                output_img = Gen_model.call(z_piror,y_condition[level],condition)
                self.outputs.append(output_img)
                output_image = sess.run(output_img,feed_dict = {z_piror:z_sample[level],y_condition[level]:y_sample_1})
                self.generator_outputs.append(output_image)
                self.condition_set.append(y_sample_1.reshape(batchsize,y_size))
            else:
                output_imgs = self.generator_outputs[level - 1]
                input_image = np.array([cv2.pyrUp(output_imgs[i,:] for i in range(batch_size))])
                self.condition_set.append(input_image)
                #input_image = tf.Variable(tf.Tensor(input_image))
                residual_img = Gen_model.call(z_piror,input_image,condition)
                self.outputs.append(residual_img)
                residual_image = sess.run(residual_img,feed_dict = {z_piror:z_sample[level],y_condition[level]:input_image})
                output_image = residual_image.data.numpy() + input_image
                self.generator_outputs.append(output_image)
                self.condition_set.append(output_image.reshape(batchsize,14 * 14 *level * level))                
        if generator:
            return self.generator_outputs
        else:
            return self.outputs,self.condition_set
        
def get_pyramid(batch_x,n_level):
    x_result = []
    length = len(batch_x)
    for x in batch_x:
        g_pyramid = []#高斯金字塔
        l_pyramid = []#拉普拉斯金字塔
        for level in range(n_level):
            if level == 0:
                g_pyramid.append(x)
            else:
                g_pyramid.append(cv2.pyrDown(g_pyramid[level - 1]))
        for level in range(n_level):
            if level == n_level - 1:
                l_pyramid.append(g_pyramid[level])
            else:
                 l_pyramid.append(g_pyramid[level] - cv2.pyrUp(g_pyramid[level + 1]))
        l_pyramid[0] = l_pyramid[0].reshape([length,28,28,1])/255.
        l_pyramid[1] = l_pyramid[1].reshape([length,14,14,1])/255.
        l_pyramid[2] = l_pyramid[2].reshape([length,7,7,1])/255.
        temp = l_pyramid[0]
        l_pyramid[0] = l_pyramid[2]
        l_pyramid[2] = temp
        x_result.append(l_pyramid)
    return x_result
def get_image(l_pyramids,n_level):
    image_result = []
    for l_pyramid in l_pyramids:
        g_pyramid = []
        for level in range(n_level):
            if level == 0:
                g_pyramid.append(l_pyramid[n_level - level - 1])
            else:
                g_pyramid.append(l_pyramid[n_level - level - 1] + cv2.pyrUP(g_pyramid[level - 1]))
        image_result.append(g_pyramid[n_level -1].reshape(28,28))
    return image_result

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

def train_lap(to_restore):
    
    g_layers = [[con2v_layer(64,3,(1,1)),con2v_layer(32,3,(1,1)),con2v_layer(1,3,(1,1))],[con2v_layer(64,6,(1,1)),con2v_layer(32,5,(1,1)),con2v_layer(1,5,(1,1))],[con2v_layer(64,7,(1,1)),con2v_layer(32,5,(2,2)),con2v_layer(1,5,(2,2))]]
    d_layers = [[con2v_layer(32,3,(1,1)),con2v_layer(64,3,(1,1)),con2v_layer(y_size,3,(1,1))],[con2v_layer(32,5,(1,1)),con2v_layer(64,5,(1,1)),con2v_layer(14*14,6,(1,1))],[con2v_layer(32,5,(2,2)),con2v_layer(64,5,(2,2)),con2v_layer(28*28,7,(1,1))]]
    LAPGAN_model = LAPGAN(n_level,g_layers,d_layers)
   
    G_losses = []
    D_losses = []
    G_optimizers = []
    D_optimizers = []
    y_reals = []
    y_generators = []
    x_reals = [tf.placeholder(tf.float32,[None,28,28,1]),tf.placeholder(tf.float32,[None,14,14,1]),tf.placeholder(tf.float32,[None,7,7,1])]
    x_generators = [tf.Variable(tf.constant(0.0,shape = [batch_size,28,28,1]), dtype = tf.float32),tf.Variable(tf.constant(0.0,shape = [batch_size,14,14,1]), dtype = tf.float32),tf.Variable(tf.constant(0.0,shape = [batch_size,7,7,1]), dtype = tf.float32)]
    #x_generators = [tf.placeholder(tf.float32,[None,28,28,1]),tf.placeholder(tf.float32,[None,14,14,1]),tf.placeholder(tf.float32,[None,7,7,1])]
    y_condition  = [tf.placeholder(tf.float32,[None,y_size]),tf.placeholder(tf.float32,[None,14*14]),tf.placeholder(tf.float32,[None,28*28])]
    optimizer = tf.train.AdamOptimizer(learn_rate)
    for level in range(n_level):
        print('hhh',level)
        y_reals.append(LAPGAN_model.Dis_models[level].call(x_reals[level],y_condition[level],condition))
        print('hhh',type(x_generators[1]))
        y_generators.append(LAPGAN_model.Dis_models[level].call(x_generators[level],y_condition[level],condition))
        G_losses.append(tf.reduce_mean(-tf.log(y_generators[level])))
        D_losses.append(tf.reduce_mean(- (tf.log(y_reals[level]) + tf.log(1 - y_generators[level]))))
        G_optimizers.append(optimizer.minimize(G_losses[level],var_list = LAPGAN_model.Gen_models[level].vars))
        D_optimizers.append(optimizer.minimize(D_losses[level],var_list = LAPGAN_model.Dis_models[level].vars))
    
    saver = tf.train.Saver(tf.global_variables())
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    if to_restore:
        chkpt_fname = tf.train.latest_checkpoint('./output')
        saver.restore(sess, chkpt_fname)

    x_sample = np.load('./phd08_data_1.npy')
    y_sample = np.load('./phd08_labels_1.npy')
    y_sample_d =one_hot(y_sample,y_size)
    y_sample_g = y_sample_d.reshape([y_sample.shape[0],1,1,y_size])
    x_sample_size = x_sample.shape[0]
    x_sample = x_sample.reshape([x_sample_size,28,28,1])
    z_sample = np.random.normal(0,1,size = (n_level,batch_size,1,1,z_size)).astype(np.float32)
    
    for epoch in range(max_epochs):
        for i in range(int(x_sample_size // batch_size)):
            x_sample_each= x_sample[batch_size * i : batch_size * (i+1)]
            x_pyramid = get_pyramid(x_sample_each,n_level)
            y_sample_1 = y_sample_g[batch_size * i : batch_size * (i+1)]/255.
            for level in range(n_level):
                x_generators,condition_set = LAPGAN_model.generate(sess,batch_size,z_sample,y_sample_1,True,False)
                _,d_l = sess.run([D_optimizers[level],D_losses[level]],feed_dict = {x_reals:x_pyramid, y_condition:condition_set})
                                              
                _,g_l = sess.run([G_optimizers[level],G_losses[level]],feed_dict = {x_reals:x_pyramid, y_condition:condition_set})
                print('epoch: %s,i:%s, d_loss: %s, g_loss: %s'%(epoch,i,d_l,g_l))
        if epoch % 5 == 0:
            x_generator = LAPGAN_model.generate(sess,batch_size,z_sample,y_sample_1,True,True)
            x_generator_image = get_image(x_generator,n_level)
            show_image(x_generator_image,'./output/lapgan_%s.jpg'% int(epoch//5))
            saver.save(sess,'./output/lapgan.model')
    sess.close()
if __name__ == '__main__':
    to_train = True
    if to_train:
        train_lap(False)
    #else:
      #  test(1)