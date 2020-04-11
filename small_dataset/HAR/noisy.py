import numpy as np 
import pandas as pd
import random 
import time 
import os 
from numpy.linalg import inv, norm, eigh
from sklearn.model_selection import train_test_split 
from scipy.optimize import minimize
import tensorflow as tf 
import tensorflow.contrib.layers as ly 
import matplotlib.pyplot as plt 
from numpy import sqrt, sin, cos, pi, exp
from scipy.integrate import quad
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from har import load_data

class syn_noise: 

    def __init__(self, arg):

        #random.seed(9)
        #np.random.seed(9)
        #tf.set_random_seed(9)

        self.arg = arg 
        self.noise_factor = self.arg.noise_term
        self.noise_scale = self.arg.noise_scale
        self.epo = self.arg.epoch
        self.batch_size = self.arg.batch_size
        self.init_weights()
        self.t_data, self.t_label, self.v_data, self.v_label = load_data()
        self.n_t_data, self.n_v_data = self.inject_noise(self.t_data, self.v_data)
        self.DNN()
        # normalizing the data
        #X_train = (X_train - np.mean(X_train)) / np.std(X_train)
        #X_test = (X_test - np.mean(X_test)) / np.std(X_test)

    def inject_noise(self, t_data, v_data): 

        t_data = t_data + self.noise_factor * np.random.laplace(0, self.noise_scale, size=(t_data.shape[0], 128, 9))
        v_data = v_data + self.noise_factor * np.random.laplace(0, self.noise_scale, size=(v_data.shape[0], 128, 9))
        return t_data, v_data


    def init_weights(self):

        self.weights = {
            'wc1': tf.Variable(tf.random_normal([1, 11, 9, 96])),
            'wc2': tf.Variable(tf.random_normal([1, 5, 96, 256])),
            'wd1': tf.Variable(tf.random_normal([32 * 32 * 2, 1000])),
            'wd2': tf.Variable(tf.random_normal([1000, 500])),
            'wd3': tf.Variable(tf.random_normal([500, 300])),
            'out': tf.Variable(tf.random_normal([300, 6]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([96])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1000])),
            'bd2': tf.Variable(tf.random_normal([500])),
            'bd3': tf.Variable(tf.random_normal([300])),
            'out': tf.Variable(tf.random_normal([6]))
        }

    def conv1d(self, x, weight, bias, stride, pad): 
        x = tf.nn.conv2d(x, weight, strides=[1, stride, 1, 1], padding=pad)
        x = tf.add(x, bias)
        return x 

    def maxpool1d(x, kernel_size, stride):
        return tf.nn.max_pool(x, ksize=[1, kernel_size, 1, 1], strides=[1, stride, 1, 1], padding='VALID')


    def Alex_net(self, input, name_1, name_2):

        with tf.variable_scope(name_1):

            #conv1 = self.conv1d(input, self.weights['wc1'], self.biases['bc1'], 4, 'SAME')
            conv1 = ly.conv2d(input, 96, kernel_size=11, stride=(4,1), padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            conv1 = ly.max_pool2d(conv1, kernel_size=3, stride=(2,1), padding='SAME')
            conv2 = ly.conv2d(conv1, 256, kernel_size=5, stride=(1,1), padding='SAME', activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer())
            #conv2 = self.bo_batch_norm(conv2, self.is_train)
            conv2 = ly.max_pool2d(conv2, kernel_size=3, stride=(2,1), padding='SAME')

            conv3 = ly.conv2d(conv2, 384, kernel_size=3, stride=(1,1), padding='SAME', activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer())
            #conv3 = self.bo_batch_norm(conv3, self.is_train)
            #conv3 = ly.max_pool2d(conv3,kernel_size=3,stride=2,padding='SAME')

            conv4 = ly.conv2d(conv3, 384, kernel_size=3, stride=(1,1), padding='SAME', activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer())
            #conv4 = self.bo_batch_norm(conv4, self.is_train)
            #conv4 = ly.max_pool2d(conv4,kernel_size=3,stride=2,padding='SAME')

            conv5 = ly.conv2d(conv4, 256, kernel_size=3, stride=(1,1), padding='SAME', activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer())
            #conv5 = self.bo_batch_norm(conv5, self.is_train)
            conv5 = ly.max_pool2d(conv5, kernel_size=3, stride=(2,1), padding='SAME')
            print(conv5)
            flat = ly.flatten(conv5)
            print(flat)

        with tf.variable_scope(name_2):

            fc1 = ly.fully_connected(flat, 4096, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            fc2 = ly.fully_connected(fc1, 4096, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            logit = ly.fully_connected(fc2, 6, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())

        return flat, logit
    '''
    def LeNet(self, input, name_1, name_2):

        with tf.variable_scope(name_1):
            conv1 = ly.conv2d(input, 6, kernel_size=5, stride=(1,1), padding='SAME', activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer())
            conv1 = ly.max_pool2d(conv1, kernel_size=2, stride=(2,1), padding='SAME')

            conv2 = ly.conv2d(conv1, 16, kernel_size=5, stride=(1,1), padding='SAME', activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer())
            conv2 = ly.max_pool2d(conv2, kernel_size=2, stride=(2,1), padding='SAME')

            #conv3 = ly.conv2d(conv2, 120, kernel_size=5, stride=1, padding='VALID', activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer())
            #conv3 = ly.max_pool2d(conv3, kernel_size=2, stride=2, padding='VALID')
            flat = ly.flatten(conv2) ### dimension 400 
            print(flat) ### dimension 400

        with tf.variable_scope(name_2):

            fc1 = ly.fully_connected(flat, 120, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            fc2 = ly.fully_connected(fc1, 84, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            logit = ly.fully_connected(fc2, 6, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())

        return flat, logit
    '''

    def LeNet(self, input, name_1, name_2):

        with tf.variable_scope(name_1):
            conv1 = tf.layers.conv1d(input, filters=6, kernel_size=5, activation= tf.nn.relu,padding='SAME')
            conv1 = tf.layers.max_pooling1d(conv1, pool_size=2, strides=2, padding='SAME')

            conv2 = tf.layers.conv1d(conv1, filters=16, kernel_size=5, padding='SAME', activation= tf.nn.relu)#, weights_initializer = tf.contrib.layers.xavier_initializer())
            conv2 = tf.layers.max_pooling1d(conv2, pool_size=2, strides=2, padding='SAME')

            #conv3 = ly.conv2d(conv2, 120, kernel_size=5, stride=1, padding='VALID', activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer())
            #conv3 = ly.max_pool2d(conv3, kernel_size=2, stride=2, padding='VALID')
            flat = ly.flatten(conv2) ### dimension 400 
            print(flat) ### dimension 400

        with tf.variable_scope(name_2):

            fc1 = ly.fully_connected(flat, 120, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            fc2 = ly.fully_connected(fc1, 84, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            logit = ly.fully_connected(fc2, 6, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())

        return flat, logit

    def DNN(self):

        self.data_p = tf.placeholder(tf.float32, shape=[None, 128, 1, 9])
        data_p = tf.reshape(self.data_p, shape=[-1, 128, 9])
        self.label_p = tf.placeholder(tf.int64, shape=[None, 6])
        #self.one_hot = tf.one_hot(self.label_p, 6)
        self.noise_p = tf.placeholder(tf.float32, shape=[None, 128, 1, 9])

        ### Tune different factor. 
        self.data_perturbed = tf.add(self.data_p,  self.noise_factor * self.noise_p)


        #self.compressing, self.logit = self.Alex_net(self.data_p, "Deep_based", "utility")   
        self.compressing, self.logit = self.LeNet(data_p, "Deep_based", "utility")   

        self.prob = tf.nn.softmax(self.logit) 

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label_p, logits=self.logit))

        theta_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Deep_based')
        theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='utility')

        op = tf.train.AdamOptimizer()
        self.opt = op.minimize(self.loss, var_list = theta_d + theta_c)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def train(self):     

        epo = 1
        for _ in range(self.epo):
            print("Epoch: {}.".format(epo))
            for i, j in self.next_batch(self.n_t_data, self.t_label, self.batch_size, shuffle=True):

                sample = i.shape[0]
                #no = np.random.multivariate_normal([0 for i in range(self.com_dim)], np.identity(self.com_dim), sample)
                #no  = np.random.normal(size=(sample, self.com_dim))
                no  = np.random.laplace(0, self.noise_scale, size=(sample, 128, 1, 9))
                feed_dict = {} 
                feed_dict[self.data_p] = i
                feed_dict[self.label_p] = j
                feed_dict[self.noise_p] = no 
                loss, _ = self.sess.run([self.loss, self.opt], feed_dict=feed_dict)

            epo += 1 
            
        acc = self.compute_acc()
        mse = self.compute_mse()
        return acc, mse 
        

    def next_batch(self, t_data, t_label, batch_size, shuffle=False):

        le = len(t_data)
        epo = le // batch_size
        leftover = le - epo * batch_size
        sup = batch_size - leftover

        if shuffle : 
            c = list(zip(t_data, t_label))
            random.shuffle(c)
            t_data , t_label = zip(*c)

        for i in range(0, le, batch_size):

            if i ==  (epo *batch_size) : 
                yield np.array(t_data[i:]).reshape(-1, 128, 1, 9) , np.array(t_label[i:])
            else : 
                yield np.array(t_data[i: i+self.batch_size]).reshape(-1, 128, 1, 9) , np.array(t_label[i: i+self.batch_size])

    def compute_mse(self):

        ### modify to self.perterb data

        temp = []
        '''
        for i, j in self.next_batch(self.n_v_data, self.v_label, self.batch_size):
            sample = i.shape[0]
            #no = np.random.multivariate_normal([0 for i in range(self.com_dim)], np.identity(self.com_dim), sample)
            #no  = np.random.normal(size=(sample, self.com_dim))
            no  = np.random.laplace(0, self.noise_scale, size=(sample, 128, 1, 9))
            feed_dict = {} 
            feed_dict[self.data_p] = i
            feed_dict[self.label_p] = j
            feed_dict[self.noise_p] = no
            perturb = self.sess.run(self.data_perturbed, feed_dict=feed_dict)

            for k in range(len(perturb)):
                temp.append(mean_squared_error(i[k].flatten(), perturb[k].flatten())*9)
        '''

        for i in range(len(self.v_data)):
            temp.append(mean_squared_error(self.v_data[i], self.n_v_data[i]))
        mse = np.mean(temp)
        print("Perturbated data MSE: {}".format(mse))
        return mse

    def compute_acc(self):

        pred = [] 

        for i, j in self.next_batch(self.n_v_data, self.v_label, self.batch_size):
            sample = i.shape[0]
            #no = np.random.multivariate_normal([0 for i in range(self.com_dim)], np.identity(self.com_dim), sample)
            #no  = np.random.normal(size=(sample, self.com_dim))
            no  = np.random.laplace(0, self.noise_scale, size=(sample, 128, 1, 9))
            feed_dict = {} 
            feed_dict[self.data_p] = i
            feed_dict[self.label_p] = j
            feed_dict[self.noise_p] = no
            prob = self.sess.run(self.prob, feed_dict=feed_dict)
            pred.append(prob)

        predict = np.concatenate(pred, axis=0)
        predict = np.argmax(predict, axis=1)
        y_true = np.argmax(self.v_label, axis=1)
        accuracy = accuracy_score(predict, y_true)
        print("Perturbated data accuracy: {}".format(accuracy))
        return accuracy





