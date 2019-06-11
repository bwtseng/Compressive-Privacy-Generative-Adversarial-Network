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
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from scipy.integrate import quad
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
### This is a theoretical solution usint the scipy optimze to solve the hard equation 
### derived from Prof.Yu and me.
class gd:

    def __init__(self, arg):

        self.arg = arg 
        self.ori_dim = self.arg.ori_dim
        self.com_dim = self.arg.com_dim
        self.samples = self.arg.samples
        self.prior_prob = self.arg.prior_prob
        self.noise_factor = self.arg.noise_term
        self.trade_off = self.arg.trade_off
        self.epo = self.arg.epoch
        self.batch_size = self.arg.batch_size
        '''
        self.seed = int(self.arg.seed)
        self.gamma = self.arg.gamma
        self.mapping_dim = self.arg.mapping_dim
        '''
        np.random.seed(9)
        random.seed(9)
        tf.set_random_seed(9)
        self.mu = np.array([[2.0 for i in range(self.ori_dim)]]).reshape(self.ori_dim, 1)
        self.t_data, self.t_label, self.v_data, self.v_label, self.cov_x, self.cov_s = self.generate_data(self.samples, self.prior_prob, self.mu)


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




    def compute_theory_acc(self, matrix_a): 

        ### put Q function here and the trace norm. 

        prior_pos = self.prior_prob
        prior_neg = 1 - self.prior_prob

        series_product = np.dot(np.dot(matrix_a, self.cov_s), matrix_a.T) 
        series_inverse = inv(series_product + (self.noise_factor)**2 * np.identity(series_product.shape[0]))
        mu_transform = 2 * np.dot(matrix_a, self.mu)

        alpha = np.dot(np.dot(mu_transform.T, series_inverse), mu_transform)
        alpha = sqrt(alpha)

        positive = prior_pos * np.round(self.Qfunc_tail(-alpha/2 + 1/alpha*np.log((prior_neg/prior_pos)))[0], 4)
        negative = prior_neg * np.round(self.Qfunc_tail(-alpha/2 - 1/alpha*np.log((prior_neg/prior_pos)))[0], 4)

        accuracy = positive + negative 
        print("Theoretical accuracy: {}".format(accuracy))
        return accuracy

    def compute_theory_mse(self, matrix_a): 

        ### BLUE 
        series_product = np.dot(np.dot(matrix_a, self.cov_x), matrix_a.T)
        series_inverse = inv(series_product +  (self.noise_factor**2)*np.identity(series_product.shape[0]))
        second_item = np.dot(self.cov_x, matrix_a.T)
        temp = np.dot(np.dot(second_item, series_inverse), second_item.T)
        value = np.trace(self.cov_x - temp) / self.ori_dim
        print("Theoretical mse: {}".format(value))

        return value

    def std_normal(self, x):
        return 1/sqrt(2*pi) * exp(-(x**2)/2)

    def Qfunc_tail(self, y):
        I = quad(self.std_normal, y,  np.inf)
        return I 

    def init_tensor(self, shape):
        initializer = tf.contrib.layers.xavier_initializer()
        return tf.Variable(initializer(shape))

    def built_model(self): 

        self.data_p = tf.placeholder(tf.float32, shape=[None, self.ori_dim])
        self.label_p = tf.placeholder(tf.int64, shape=[None])
        self.one_hot = tf.one_hot(self.label_p, 2)
        self.noise_p = tf.placeholder(tf.float32, shape=[None, self.com_dim])
        #self.emb_p = tf.placeholder(tf.float32, shape=[None, self.com_dim])
        #self.kernel_map = self.RFF_map(self.emb_p, self.seed, self.gamma, self.com_dim, self.ori_dim)
        with tf.variable_scope('privatizer'):
            #compressing = self.fs_layer(self.data_p, 20)
            #self.compressing = self.fs_layer(self.data_p, self.com_dim)
            #self.compressing = self.fs_layer(self.data_p, 16)
            self.compressing = self.fs_layer(self.data_p, self.com_dim)
            #self.compressing = self.fs_layer(self.compressing, self.)

        self.theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='privatizer')

        with tf.variable_scope('classifier'):

            #pre_logit = self.fs_layer(self.compressing, 8)
            #pre_logit = self.fs_layer(pre_logit, 4)
            self.logit = self.fs_layer(self.compressing, 2)
            self.prob = tf.nn.softmax(self.logit)

        self.theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')

        with tf.variable_scope('adversary_nn') as scope:
            self.recon_nn = self.fs_layer(self.compressing, self.ori_dim)

        self.theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')
        self.theta_r = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='adversary_nn')
        self.theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='privatizer')


        self.loss_uti = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= self.one_hot, logits= self.logit))
        self.loss_r = tf.losses.mean_squared_error(self.data_p, self.recon_nn) 

        self.c_op = tf.train.AdamOptimizer()
        self.c_opt = self.c_op.minimize(self.loss_uti, var_list=self.theta_c)

        self.r_op = tf.train.AdamOptimizer()
        self.r_opt = self.r_op.minimize(self.loss_r, var_list=self.theta_r)

    def train(self):

        ### should examine this function.

        self.mu_p = tf.placeholder(tf.float32, shape=[self.ori_dim, 1])
        self.cov_s_p = tf.placeholder(tf.float32, shape=[self.ori_dim, self.ori_dim])
        self.cov_x_p = tf.placeholder(tf.float32, shape=[self.ori_dim, self.ori_dim])
        self.cov_n_p = tf.placeholder(tf.float32, shape=[self.com_dim, self.com_dim])


        '''
        self.data_p = tf.placeholder(tf.float32, shape=[None, self.ori_dim])
        self.emb_p = tf.placeholder(tf.float32, shape=[None, self.com_dim])

        self.privatizer_weight_p = tf.placeholder(tf.float32, shape=[self.ori_dim, self.com_dim])
        self.krr_weights_p = tf.placeholder(tf.float32, shape=[self.mapping_dim, self.ori_dim])
        self.lrr_weights_p = tf.placeholder(tf.float32, shape=[self.com_dim, self.ori_dim])

        self.emb_map = self.RFF_map(self.emb_p, self.seed, self.gamma, self.com_dim, self.ori_dim)
        self.krr_reco = self.adversary_krr(self.emb_map)
        self.lrr_reco = self.adversary_lrr(self.emb_p)

        self.theta_krr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='adversary_krr')
        self.theta_lrr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='adversary_lrr')

        self.assign_op = []
        self.assign_op.append(self.theta_krr[0].assign(self.krr_weights_p))
        self.assign_op.append(self.theta_lrr[0].assign(self.lrr_weights_p))

        with tf.variable_scope("Privatizer"): 

            self.compressing = self.fs_layer(self.data_p, self.com_dim)
            #self.compressing = self.fs_layer(compressing, self.com_dim)
        self.theta_pri = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Privatizer')  
        self.assign_pri = []         
        self.assign_pri.append(self.theta_pri[0].assign(self.privatizer_weight_p))

        with tf.variable_scope('white_box'):
            white_box = self.fs_layer(self.emb_p, 20)
            white_box = self.fs_layer(white_box, 24)
            white_box = self.fs_layer(white_box, 28)
            self.recon_white_box = self.fs_layer(white_box, self.ori_dim)

        uti_update_white_box = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='white_box')
        with tf.control_dependencies(uti_update_white_box):
            self.theta_white_box = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='white_box')
            self.loss_white_box = tf.losses.mean_squared_error(self.data_p, self.recon_white_box)
            self.opt_white_box = tf.train.AdamOptimizer(0.001).minimize(self.loss_white_box, var_list= self.theta_white_box)
        '''


        self.A_weights = tf.placeholder(tf.float32, shape=[self.ori_dim, self.com_dim])

        A = self.init_tensor((self.com_dim, self.ori_dim))
        A_T = tf.transpose(A)

        ### Q function term. (with respect to mu), in the case that both they are the same mean value. 
        transform_mu = 2 * tf.matmul(A, self.mu_p)
        transform_mu_T = tf.transpose(transform_mu)
        transform_cov_s = tf.matmul(tf.matmul(A, self.cov_s_p), A_T)
        add_noise_s = tf.add(transform_cov_s, (self.noise_factor**2) * self.cov_n_p)
        inverse_term_s = tf.linalg.inv(add_noise_s)
        second_term = tf.matmul(tf.matmul(transform_mu_T, inverse_term_s), transform_mu)

        ### MSE term, it's may possible not to work.
        transform_cov_x = tf.matmul(tf.matmul(A, self.cov_x_p), A_T)
        add_noise_x =  tf.add(transform_cov_x, (self.noise_factor**2) * self.cov_n_p)
        inverse_term_x = tf.linalg.inv(add_noise_x)
        side_term = tf.matmul(A, self.cov_x_p)
        side_term_T = tf.transpose(side_term)
        first_term = tf.linalg.trace(self.cov_x_p) - tf.linalg.trace(tf.matmul(tf.matmul(side_term_T, inverse_term_x), side_term))
        ### to control the service performance balance between privacy and utility.
        final_func = - (self.trade_off * second_term + first_term)
        ### Scipy optimizer in tensorflow 
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
        final_func, method='SLSQP', options={'maxiter': 100000})
        #'SLSQP', 'L-BFGS-B'

        self.built_model()
        self.assign_op = []
        self.assign_op.append(tf.assign(self.theta_g[0], self.A_weights))
        print(self.assign_op)
        ## May consider to use the Gradient Descent. 
        op = tf.train.AdamOptimizer()
        opt = op.minimize(final_func)




        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())        
        feed_dict = {}
        feed_dict[self.mu_p] = self.mu
        feed_dict[self.cov_s_p] = self.cov_s
        feed_dict[self.cov_x_p] = self.cov_x
        feed_dict[self.cov_n_p] = 0.001 * np.identity(self.com_dim)
        optimizer.minimize(self.sess, feed_dict=feed_dict) #,options={'maxiter': 1000})
        '''
        for i in range(200):
            self.sess.run(opt, feed_dict=feed_dict)
            #print(self.sess.run(A))
        '''

        #print(self.sess.run(A))
        #print(self.sess.run(A))

        matrix_A = self.sess.run(A)
        self.sess.run(self.assign_op, feed_dict={self.A_weights:matrix_A.T})
        theory_acc = self.compute_theory_acc(matrix_A)
        theory_mse = self.compute_theory_mse(matrix_A)


        for _ in range(self.epo):
            for i, j in self.next_batch(self.t_data, self.t_label, self.batch_size):
                feed_dict = {} 
                feed_dict[self.data_p] = i
                feed_dict[self.label_p] = j
                self.sess.run([self.r_opt, self.c_opt], feed_dict=feed_dict)


        gd_acc = self.evalute_utility(self.v_data, self.v_label)
        gd_mse = self.evalute_privacy(self.v_data, self.v_label)

        return theory_acc, theory_mse, gd_acc, gd_mse

        #self.sess.run(self.assign_pri, feed_dict={self.privatizer_weight_p: self.sess.run(A)})


    def evalute_privacy(self, v_data, label):


        mse_list = []
        for i, j in self.next_batch(v_data, label, self.batch_size):
            feed_dict = {} 
            feed_dict[self.data_p] = i
            feed_dict[self.label_p] = j
            reco = self.sess.run(self.recon_nn, feed_dict=feed_dict)
            for k in range(len(reco)):
                mse_list.append(mean_squared_error(i[k], reco[k]))
                    
        mean_mse = np.mean(mse_list)
        print("white_box attack MSE (NN): {}.".format(mean_mse))
        return mean_mse


    def evalute_utility(self, v_data, v_label):

        prob_list = []
        #for i in range(len(data_list)):
        for i, j in self.next_batch(v_data, v_label, self.batch_size):
            feed_dict = {} 
            feed_dict[self.data_p] = i
            feed_dict[self.label_p] = j
            reco = self.sess.run(self.prob, feed_dict=feed_dict)
            prob_list.append(reco)


        predict = np.concatenate(prob_list, axis=0)
        predict = np.argmax(predict, axis=1)


        TP = [] 
        NP = []
        for i, j in zip(v_label, predict):
            #print(i)
            if i == 1 : 
                if i == j : 
                    TP.append(1)
            if i == 0 : 
                if i == j : 
                    NP.append(1)

        acc = (len(TP) + len(NP)) / len(v_data)
        accuracy = accuracy_score(predict, np.array(v_label))
        print("Gradient descent accuracy: {}".format(accuracy))
        print("Gradient descent accuracy (TP and NP): {}".format(acc))
        return accuracy


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


    def KRR_close_form(self, emb_matrix, train_matrix):
        # Use the random fourier transform to approximate the RBF kernel 
        # Note that the training data is too large so that we use the intrinsic space mapping 
        # And use the tensorflow conrtrib package to get the RFF mapping rather than hand crafting
        # More information refers to https://github.com/hichamjanati/srf  

        rau = 0.001
        mu = np.mean(emb_matrix, axis=0)
        emb_matrix = emb_matrix - mu

        '''
        emb_matrix = emb_matrix.T 
        s = np.dot(emb_matrix, emb_matrix.T)
        a,b = s.shape
        identity = np.identity(a)
        s_inv = np.linalg.inv(s + rau * np.identity(a))
        '''
        train_norm = train_matrix - train_mu
        #weights = np.dot(np.dot(s_inv,emb_matrix), train_norm)
        #weights = np.dot(np.dot(s_inv, emb_matrix), train_matrix)
        clf = Ridge(alpha=0.001, fit_intercept=False, random_state=9)
        clf.fit(emb_matrix, train_norm)
        weights = clf.coef_
        print("Shape of KRR weights: {}.".format(weights.shape))
        return weights.T


    def LRR_close_form(self, emb_matrix, train_matrix):
        mu = np.mean(emb_matrix, axis=0)
        #print("Shape of mu: {}.".format(mu.shape))
        emb_matrix = emb_matrix - mu
        '''
        emb_matrix = emb_matrix.T
        rau = 0.001
        s = np.dot(emb_matrix, emb_matrix.T)
        h,w = s.shape
        s_inv = np.linalg.inv(s + rau*np.identity(h))
        '''
        train_norm = train_matrix - train_mu
        clf = Ridge(alpha=0.001, fit_intercept=False, random_state=9)
        clf.fit(emb_matrix, train_norm)
        weights = clf.coef_
        #weights = np.dot(np.dot(s_inv, emb_matrix), train_norm)
        #weights = np.dot(np.dot(s_inv, emb_matrix), train_matrix)
        print("Shape of LRR weights: {}.".format(weights.shape))
        return weights.T



















