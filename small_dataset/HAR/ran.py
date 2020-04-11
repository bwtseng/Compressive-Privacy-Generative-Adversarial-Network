from har import load_data
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
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import laplacian_kernel
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from sklearn.utils import check_random_state
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes

import math

class RAN:
    def __init__(self, arg):

        self.arg = arg 
        self.trade_off = self.arg.trade_off
        self.ori_dim = 128*1*9
        self.epo = self.arg.epoch
        self.com_dim = 512
        self.mapping_dim = self.arg.mapping_dim
        self.gamma = self.arg.gamma
        self.seed = self.arg.seed
        self.batch_size = self.arg.batch_size
        self.t_data, self.t_label, self.v_data, self.v_label = load_data()
        self.RAN()


    def deconv(self, x, units, kernel_size, stride, activetion, pad): 
        return ly.conv2d_transpose(x, units, kernel_size=kernel_size, stride=(stride, 1), activation_fn=activetion, padding=pad, weights_initializer=tf.contrib.layers.xavier_initializer())

    def conv(self, x, kernel_size, stride, activetion, pad): 
        return ly.conv2d(x, kernel_size=k, stride=(stride, 1), activation_fn=activetion, padding=pad, weights_initializer=tf.contrib.layers.xavier_initializer())

    def Alex_net(self, input, name_1, name_2):
        ### This is one dimension version. 
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


    def adversary_krr(self, kernel_vec, reuse=False):
        #final_latent = tf.concat([latent, latent_1], axis=1)
        with tf.variable_scope('adversary_krr') as scope:  
            if reuse: 
                scope.reuse_variables()
            recontruction = ly.fully_connected(kernel_vec, self.ori_dim, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = None)
        #return tf.reshape(recontruction, shape=[-1, 175, 175, 3])
        return tf.reshape(recontruction, shape=[-1, 128, 1, 9])


    def adversary_lrr(self, compressive_data, reuse=False):
        #final_latent = tf.concat([latent, latent_1], axis=1)
        with tf.variable_scope('adversary_lrr') as scope:  
            if reuse: 
                scope.reuse_variables()
            recontruction = ly.fully_connected(compressive_data, self.ori_dim, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = None)
        #return tf.reshape(recontruction, shape=[-1, 175, 175, 3])
        return tf.reshape(recontruction, shape=[-1, 128, 1, 9])


    def RFF_map(self, input_tensor, seed, stddev, input_shape, output_dim):
        #input_tensor = tf.concat([input_tensor_1, input_tensor_2], axis=1)
        #print("Information that the adversary can get: {}".format(input_tensor))

        #random_state = check_random_state(seed)
        gamma = stddev
        omega_matrix_shape = [input_shape, output_dim]
        bias_shape = [output_dim]

        omega_matrix = constant_op.constant(np.sqrt(2 * gamma) *
           np.random.normal(size=omega_matrix_shape), dtype=dtypes.float32)

        bias = constant_op.constant(
            np.random.uniform(
            0.0, 2 * np.pi, size=bias_shape), dtype=dtypes.float32)

        x_omega_plus_bias = math_ops.add(
            math_ops.matmul(input_tensor, omega_matrix), bias)

        return math.sqrt(2.0 / output_dim) * math_ops.cos(x_omega_plus_bias)


    def RAN(self):

        self.data_p = tf.placeholder(tf.float32, shape=[None, 128, 1, 9])
        self.label_p = tf.placeholder(tf.int64, shape=[None, 6])
        self.emb_p = tf.placeholder(tf.float32, shape=[None, self.com_dim])

        #noise_p = tf.placeholder(tf.float32, shape=[None, self.com_dim])

        with tf.variable_scope('white_box'):

            reshape = tf.reshape(self.emb_p, shape=[-1, 4, 1, 128])
            deconv_white_obx = self.deconv(reshape, 64, 3, 4, tf.nn.relu, 'SAME')
            deconv_white_obx = self.deconv(deconv_white_obx, 32, 3, 2, tf.nn.relu, 'SAME')
            deconv_white_obx = self.deconv(deconv_white_obx, 16, 3, 2, tf.nn.relu, 'SAME')
            self.deconv_white_box = self.deconv(deconv_white_obx, 9, 3, 2, None, 'SAME')
            
        uti_update_white_box = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='white_box')

        with tf.control_dependencies(uti_update_white_box):
            self.theta_white_box = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='white_box')
            self.loss_white_box = tf.losses.mean_squared_error(self.data_p, self.deconv_white_box)
            self.opt_white_box = tf.train.AdamOptimizer().minimize(self.loss_white_box, var_list= self.theta_white_box)

        #self.compressing, self.logit = self.Alex_net(self.data_p, "Encoder", "Classifier")
        self.compressing, self.logit = self.LeNet(self.data_p, "Encoder", "Classifier")
        self.prob = tf.nn.softmax(self.logit)

        self.krr_weights_p = tf.placeholder(tf.float32, shape=[self.mapping_dim, self.ori_dim])
        self.lrr_weights_p = tf.placeholder(tf.float32, shape=[self.com_dim, self.ori_dim])

        self.emb_map = self.RFF_map(self.emb_p, self.seed, self.gamma, self.com_dim, self.mapping_dim)
        self.krr_reco = self.adversary_krr(self.emb_map)
        self.lrr_reco = self.adversary_lrr(self.emb_p)

        self.theta_krr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='adversary_krr')
        self.theta_lrr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='adversary_lrr')

        self.assign_op = []
        self.assign_op.append(self.theta_krr[0].assign(self.krr_weights_p))
        self.assign_op.append(self.theta_lrr[0].assign(self.lrr_weights_p))

        with tf.variable_scope("Decoder"):
            reshape_d = tf.reshape(self.compressing, shape=[-1, 4, 1, 128])
            deconv_d = self.deconv(reshape_d, 64, 3, 4, tf.nn.relu, 'SAME')
            deconv_d = self.deconv(deconv_d, 32, 3, 2, tf.nn.relu, 'SAME')
            deconv_d = self.deconv(deconv_d, 16, 3, 2, tf.nn.relu, 'SAME')
            self.deconv_d = self.deconv(deconv_d, 9, 3, 2, None, 'SAME')

            #self.deconv_d = self.deconv(reshape, 3, 2, None, 'SAME')

        self.loss_pri = tf.losses.mean_squared_error(self.data_p, self.deconv_d) 
        self.loss_uti = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= self.label_p, logits= self.logit))

        self.loss_g = self.trade_off * self.loss_uti - (1 - self.trade_off) * self.loss_pri
        
        theta_e = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encoder')

        theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Classifier')

        theta_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decoder')

        uti_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        scope = ["Encoder", "Classifier", "Decoder"]
        uti_update = []
        for i in scope : 
            uti_update += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=i)
        with tf.control_dependencies(uti_update):
            e_c_op = tf.train.AdamOptimizer(0.001)
            self.e_c_opt = e_c_op.minimize(self.loss_uti, var_list = theta_e + theta_c)

            d_op = tf.train.AdamOptimizer(0.001)
            self.d_opt = d_op.minimize(self.loss_pri, var_list = theta_d)

            g_op = tf.train.AdamOptimizer(0.001)
            self.g_opt = g_op.minimize(self.loss_g, var_list= theta_e + theta_c)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=5)

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

    def train(self):

        epo = 1
        for _ in range(self.epo):
            citers = 3 
            print("Epoch {} starts.".format(epo))
            for i, j in self.next_batch(self.t_data, self.t_label, self.batch_size, shuffle=True):
                sample = i.shape[0]
                #no  = np.random.normal(size=(sample, self.com_dim))

                feed_dict = {} 
                feed_dict[self.data_p] = i
                feed_dict[self.label_p] = j

                for _ in range(citers):
                    _ = self.sess.run([self.e_c_opt], feed_dict=feed_dict)
                    _ = self.sess.run([self.d_opt], feed_dict=feed_dict)
                #_ = self.sess.run([self.e_c_opt], feed_dict=feed_dict)
                _ = self.sess.run([self.g_opt], feed_dict=feed_dict)

            epo += 1
        acc = self.prediction_and_accuracy(self.v_data, self.v_label)
        mse, mse_lrr, mse_krr = self.evalute_privacy(self.t_data, self.v_data, self.v_label)
        return acc, mse, mse_lrr, mse_krr

    def prediction_and_accuracy(self, data, label):

        pred = [] 
        for i, j in self.next_batch(data, label, self.batch_size):

            sample = i.shape[0]
            #no  = np.random.laplace(size=(sample, self.com_dim))
            feed_dict = {} 
            feed_dict[self.data_p] = i
            #feed_dict[self.noise_p] = self.factor * no
            prob = self.sess.run(self.prob, feed_dict=feed_dict)
            pred.append(prob)

        predict = np.concatenate(pred, axis=0)
        predict = np.argmax(predict, axis=1)
        y_true = np.argmax(self.v_label, axis=1)

        accuracy = accuracy_score(predict, y_true)       
        print("Deep features Accuracy: {}".format(accuracy))
        return accuracy

    def get_train_matrix(self): 

        real_list = []
        count = 0 
        temp = [] 
        for i,j in self.next_batch(self.t_data, self.t_label, self.batch_size): 
            #k = imresize(self.plot(imread(i)),(175,175))
            #k = (k/127.5) -1
            temp.append(i.reshape(-1, 128*9))
        train_matrix = np.concatenate(temp, axis=0)
        print(train_matrix.shape)
        print('Successfully get flatted train matrix !!!!')     
        return train_matrix
        
    def evalute_privacy(self, data, v_data, label):

        data_list = []
        emb_list = []
        map_list = []
        epo = 1
        
        for _ in range(40):
            for i, j in self.next_batch(data, label, self.batch_size):

                sample = i.shape[0]
                #no  = np.random.laplace(size=(sample, self.com_dim))
                emb = self.sess.run(self.compressing, feed_dict = {self.data_p:i})
                rff_map = self.sess.run(self.emb_map, feed_dict = {self.emb_p:emb}) 
                _ = self.sess.run(self.opt_white_box, feed_dict={self.emb_p:emb, self.data_p:i})
                if epo == 1 : 
                    data_list.append(i)
                    emb_list.append(emb)
                    map_list.append(rff_map)
            epo +=1

        emb = np.concatenate(emb_list, axis=0)
        rff_map = np.concatenate(map_list, axis=0)

        train_matrix = self.get_train_matrix()
        feed_dict = {}
        feed_dict[self.krr_weights_p] = self.KRR_close_form(rff_map, train_matrix)
        feed_dict[self.lrr_weights_p] = self.LRR_close_form(emb, train_matrix)
        self.sess.run(self.assign_op, feed_dict=feed_dict)

        mse_list = []
        mse_list_lrr = []
        mse_list_krr = []


        for i, j in self.next_batch(v_data, label, self.batch_size):
            sample = i.shape[0]
            emb = self.sess.run(self.compressing, feed_dict = {self.data_p:i})
            #rff_map = self.sess.run(self.emb_p, feed_dict = {self.emb_p:emb})
            feed_dict = {}
            feed_dict[self.emb_p] = emb 
            reco, reco_lrr, reco_krr = self.sess.run([self.deconv_white_box, self.lrr_reco, self.krr_reco], feed_dict=feed_dict)
            for k in range(len(reco)):
                mse_list.append(mean_squared_error(i[k].flatten(), reco[k].flatten())*9)
                mse_list_lrr.append(mean_squared_error(i[k].flatten(), reco_lrr[k].flatten())*9)
                mse_list_krr.append(mean_squared_error(i[k].flatten(), reco_krr[k].flatten())*9)

        mean_mse = np.mean(mse_list)
        mean_mse_lrr = np.mean(mse_list_lrr)
        mean_mse_krr = np.mean(mse_list_krr)

        print("white_box attack MSE (NN): {}.".format(mean_mse))
        print("white_box attack MSE (LRR): {}.".format(mean_mse_lrr))
        print("white_box attack MSE (KRR): {}.".format(mean_mse_krr))
        return mean_mse, mean_mse_lrr, mean_mse_krr

    def KRR_close_form(self, emb_matrix, train_matrix):
        # Use the random fourier transform to approximate the RBF kernel 
        # Note that the training data is too large so that we use the intrinsic space mapping 
        # And use the tensorflow conrtrib package to get the RFF mapping rather than hand crafting
        # More information refers to https://github.com/hichamjanati/srf  

        rau = 1
        mu = np.mean(emb_matrix, axis=0)
        emb_matrix = emb_matrix - mu

        emb_matrix = emb_matrix.T 
        s = np.dot(emb_matrix, emb_matrix.T)
        a,b = s.shape
        identity = np.identity(a)
        s_inv = np.linalg.inv(s + rau * np.identity(a))
        train_mu = np.mean(train_matrix, axis=0)
        train_norm = train_matrix - train_mu
        weights = np.dot(np.dot(s_inv,emb_matrix), train_norm)
        #weights = np.dot(np.dot(s_inv, emb_matrix), train_matrix)
        #clf = Ridge(alpha=1, fit_intercept=False, random_state=9)
        #clf.fit(emb_matrix, train_norm)
        #weights = clf.coef_
        print("Shape of KRR weights: {}.".format(weights.shape))
        return weights


    def LRR_close_form(self, emb_matrix, train_matrix):
        mu = np.mean(emb_matrix, axis=0)
        #print("Shape of mu: {}.".format(mu.shape))
        emb_matrix = emb_matrix - mu
        
        emb_matrix = emb_matrix.T
        rau = 0.001
        s = np.dot(emb_matrix, emb_matrix.T)
        h,w = s.shape
        s_inv = np.linalg.inv(s + rau*np.identity(h))
        train_mu = np.mean(train_matrix, axis=0)
        train_norm = train_matrix - train_mu
        #clf = Ridge(alpha=0.01, fit_intercept=False, random_state=9)
        #clf.fit(emb_matrix, train_norm)
        #weights = clf.coef_
        weights = np.dot(np.dot(s_inv, emb_matrix), train_norm)
        #weights = np.dot(np.dot(s_inv, emb_matrix), train_matrix)
        print("Shape of LRR weights: {}.".format(weights.shape))
        return weights










    