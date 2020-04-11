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
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes

class syn_noise: 

    def __init__(self, arg):

        #random.seed(9)
        #np.random.seed(9)
        #tf.set_random_seed(9)

        self.arg = arg 
        self.ori_dim = self.arg.ori_dim
        self.com_dim = self.arg.com_dim
        self.samples = self.arg.samples
        self.noise_factor = self.arg.noise_term
        self.batch_size = self.arg.batch_size
        self.epo = self.arg.epoch
        self.mu = np.array([[2.0 for i in range(self.ori_dim)]]).reshape(self.ori_dim, 1)
        self.prior_prob = self.arg.prior_prob
        self.t_data, self.t_label, self.v_data, self.v_label, self.cov_x, self.cov_s = self.generate_data(self.samples, self.prior_prob, self.mu)
        #print(self.t_data)
        #print(self.v_data)
        self.DNN()


    def generate_label(self, n_samples, prior_prob):

        positive_samples = n_samples * prior_prob
        negative_samples = n_samples * (1 - prior_prob)
        t_label = [1 for i in range(int(positive_samples))] + [0 for i in range(int(negative_samples))]

        #positive_samples_val = 2000 * prior_prob
        #negative_samples_val = 2000 * (1 - prior_prob)

        positive_samples_val = 1000
        negative_samples_val = 1000

        v_label = [1 for i in range(int(positive_samples_val))] + [0 for i in range(int(negative_samples_val))]

        return t_label, v_label 

    def generate_data(self, n_samples, prior_prob, mu):

        ### mu is a column vector. 
        positive_samples = n_samples * prior_prob
        negative_samples = n_samples * (1 - prior_prob)
        label = [1 for i in range(int(positive_samples))] + [0 for i in range(int(negative_samples))]


        #C = np.array([[2.0, 0.3], [0.3, 3.0]])

        C = np.random.rand(self.ori_dim, self.ori_dim)

        mu_pos = [2 for i in range(self.ori_dim)] 
        mu_neg = [-2 for i in range(self.ori_dim)]

        positive_matrix = np.random.multivariate_normal(mu_pos, C.T.dot(C), int(positive_samples))
        negative_matrix = np.random.multivariate_normal(mu_neg, C.T.dot(C), int(negative_samples))

        positive_val = random.sample([i for i in range(positive_matrix.shape[0])], 1000)
        negative_val = random.sample([i for i in range(negative_matrix.shape[0])], 1000)

        train_matrix = np.concatenate([positive_matrix, negative_matrix], axis=0)

        t_data = []
        v_data_pos = []
        v_data_neg = []

        t_label = []
        v_label = []

        for i in range(len(positive_matrix)):
            if i in positive_val : 
                v_data_pos.append(positive_matrix[i])
                v_label.append(1)
            else : 
                t_data.append(positive_matrix[i])
                t_label.append(1)

        cov_s_1 = np.cov(np.array(v_data_pos).T, ddof=0)

        for i in range(len(negative_matrix)):
            if i in negative_val : 
                v_data_neg.append(negative_matrix[i])
                v_label.append(0)
            else : 
                t_data.append(negative_matrix[i])
                t_label.append(0)


        cov_s_2 = np.cov(np.array(v_data_neg).T, ddof=0)
        cov_s = (cov_s_1 + cov_s_2 )/ 2 


        v_data = v_data_pos + v_data_neg

        assert len(train_matrix) == (len(t_data) + len(v_data))

        cov_x = np.cov(np.array(v_data).T, ddof=0)
        #return positive_matrix, negative_matrix, positive_matrix_val, negative_matrix_val, train_matrix, cov_x, cov_s
        return np.array(t_data).reshape(-1, self.ori_dim), t_label, np.array(v_data).reshape(-1, self.ori_dim), v_label, cov_x, cov_s


    def fs_layer(self, input, output_dim, bias = False):

        if bias :  
            return ly.fully_connected(input, output_dim, activation_fn = None, weights_initializer=tf.contrib.layers.xavier_initializer())#, biases_initializer = None)
        else : 
            return ly.fully_connected(input, output_dim, activation_fn = None, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = None)


    def DNN(self):

        self.data_p = tf.placeholder(tf.float32, shape=[None, self.ori_dim])
        self.label_p = tf.placeholder(tf.int64, shape=[None])
        self.one_hot = tf.one_hot(self.label_p, 2)
        self.noise_p = tf.placeholder(tf.float32, shape=[None, self.ori_dim])
        self.noise_p = tf.placeholder(tf.float32, shape=[None ,self.ori_dim])

        ### Tune different factor. 
        self.data_perturbed = tf.add(self.data_p, self.noise_factor * self.noise_p)

        with tf.variable_scope("Deep_based"):
            compressing = self.fs_layer(self.data_perturbed, 16)
            compressing = self.fs_layer(compressing, 8)
            pre_logit = self.fs_layer(compressing, 4)
            self.logit = self.fs_layer(pre_logit, 2)    
            self.prob = tf.nn.softmax(self.logit) 

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot, logits=self.logit))

        theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Deep_based')

        op = tf.train.AdamOptimizer()
        self.opt = op.minimize(self.loss, var_list = theta)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()


    def train(self):

        for _ in range(self.epo):
            for i, j in self.next_batch(self.t_data, self.t_label, self.batch_size, shuffle=True):

                sample = i.shape[0]
                #no  = np.random.normal(size=(sample, self.com_dim))
                no  = np.random.laplace(0, 0.5, size=(sample, self.ori_dim))
                feed_dict = {} 
                feed_dict[self.data_p] = i.reshape(-1, self.ori_dim)
                feed_dict[self.label_p] = j.reshape(-1)
                feed_dict[self.noise_p] = no 
                loss, _ = self.sess.run([self.loss, self.opt], feed_dict=feed_dict)

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
                yield np.array(t_data[i:]) , np.array(t_label[i:])
            else : 
                yield np.array(t_data[i: i+self.batch_size]) , np.array(t_label[i: i+self.batch_size])

    def compute_mse(self):

        ### modify to self.perterb data

        temp = []
        for i, j in self.next_batch(self.v_data, self.v_label, self.batch_size):
            sample = i.shape[0]
            #no = np.random.multivariate_normal([0 for i in range(self.com_dim)], np.identity(self.com_dim), sample)
            #no  = np.random.normal(size=(sample, self.com_dim))
            no  = np.random.laplace(0, 0.5, size=(sample, self.ori_dim))
            feed_dict = {} 
            feed_dict[self.data_p] = i.reshape(-1, self.ori_dim)
            feed_dict[self.label_p] = j.reshape(-1)
            feed_dict[self.noise_p] = no

            perturb = self.sess.run(self.data_perturbed, feed_dict=feed_dict)
            for k in range(len(perturb)):
                temp.append(mean_squared_error(i[k], perturb[k])*self.ori_dim)

        mse = np.mean(temp)
        #mse = np.sum(temp)
        print("Perturbated data MSE: {}".format(mse))
        return mse

    def compute_acc(self):

        pred = [] 

        for i, j in self.next_batch(self.v_data, self.v_label, self.batch_size):
            sample = i.shape[0]
            #no = np.random.multivariate_normal([0 for i in range(self.com_dim)], np.identity(self.com_dim), sample)
            #no  = np.random.normal(size=(sample, self.com_dim))
            no  = np.random.laplace(0, 0.5, size=(sample, self.ori_dim))
            feed_dict = {} 
            feed_dict[self.data_p] = i.reshape(-1, self.ori_dim)
            feed_dict[self.label_p] = j.reshape(-1)
            feed_dict[self.noise_p] = no
            prob = self.sess.run(self.prob, feed_dict=feed_dict)
            pred.append(prob)

        predict = np.concatenate(pred, axis=0)
        predict = np.argmax(predict, axis=1)
        accuracy = accuracy_score(predict, np.array(self.v_label))
        print("Perturbated data accuracy: {}".format(accuracy))
        return accuracy
