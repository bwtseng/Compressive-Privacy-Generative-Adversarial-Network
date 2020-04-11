import numpy as np 
import pandas as pd
import random 
import time 
import os 
from numpy.linalg import inv, norm, eigh
from sklearn.model_selection import train_test_split 
from scipy.optimize import minimize
from numpy import sqrt, sin, cos, pi, exp
import tensorflow as tf 
import tensorflow.contrib.layers as ly 
import matplotlib.pyplot as plt 
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from scipy.integrate import quad

class hybrid_CPGAN:
    def __init__(self, arg):

        self.arg = arg
        #np.random.seed(9)
        #random.seed(9)
        #tf.set_random_seed(9)
        self.ori_dim = self.arg.ori_dim
        self.com_dim = self.arg.com_dim
        self.samples = self.arg.samples
        self.citer = self.arg.citer
        self.trade_off = self.arg.trade_off
        self.pca_dim = self.arg.pca_dim
        self.mapping_dim = self.arg.mapping_dim
        self.batch_size = self.arg.batch_size
        self.noise_factor = self.arg.noise_term
        self.epo = self.arg.epoch
        #self.mode = self.arg.mode  
        self.seed = self.arg.seed
        self.gamma = self.arg.gamma
        self.mu = np.array([[2 for i in range(self.ori_dim)]]).reshape(self.ori_dim, 1)
        self.prior_prob = self.arg.prior_prob
        self.t_data, self.t_label, self.v_data, self.v_label, self.cov_x, self.cov_s = self.generate_data(self.samples, self.prior_prob, self.mu)
        self.CPGAN()

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


    def RFF_map(self, input_tensor, seed, stddev, input_shape, output_dim):
        #input_tensor = tf.concat([input_tensor_1, input_tensor_2], axis=1)
        #print("Information that the adversary can get: {}".format(input_tensor))

        #random_state = check_random_state(seed)
        gamma = stddev
        omega_matrix_shape = [input_shape, output_dim]
        bias_shape = [output_dim]

        """
        This is the tensorflow version RFF mapping, but I refer to the scikit-learn version.!!!!
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        np.random.seed(9)
        self._stddev = stddev
        omega_matrix_shape = [self.arg.dim*2, output_dim]
        bias_shape = [output_dim]

        omega_matrix = constant_op.constant(
            np.random.normal(
            scale=1.0 / self._stddev, size=omega_matrix_shape),
            dtype=dtypes.float32)

        bias = constant_op.constant(
            np.random.uniform(
            low=0.0, high=2 * np.pi, size=bias_shape),
            dtype=dtypes.float32)

        x_omega_plus_bias = math_ops.add(
            math_ops.matmul(input_tensor, omega_matrix), bias)

        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        """

        omega_matrix = constant_op.constant(np.sqrt(2 * gamma) *
           np.random.normal(size=omega_matrix_shape), dtype=dtypes.float32)

        bias = constant_op.constant(
            np.random.uniform(
            0.0, 2 * np.pi, size=bias_shape), dtype=dtypes.float32)

        x_omega_plus_bias = math_ops.add(
            math_ops.matmul(input_tensor, omega_matrix), bias)

        return math.sqrt(2.0 / output_dim) * math_ops.cos(x_omega_plus_bias)

    def adversary_lrr(self, final_latent, reuse=False):
        # If deduct mu, then.....
        with tf.variable_scope('adversary_lrr') as scope:  
            if reuse: 
                scope.reuse_variables()
            recontruction = ly.fully_connected(final_latent, self.ori_dim, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = None)
        #return tf.reshape(recontruction, shape=[-1, 175, 175, 3])
        return recontruction


    def adversary_krr(self, kernel_map, reuse=False):
        # If deduct mu, then.....
        with tf.variable_scope('adversary_krr') as scope:  
            if reuse: 
                scope.reuse_variables()
            recontruction = ly.fully_connected(kernel_map, self.ori_dim, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = None)
        return recontruction


    def adversary_nn(self, final_latent, reuse=False):

        with tf.variable_scope('adversary_nn') as scope:
            if reuse:
                scope.reuse_variables()
            #upsampling = self.fs_layer(final_latent, 20)
            #upsampling = self.fs_layer(upsampling, 24)
            #upsampling = self.fs_layer(upsampling, 28)
            #upsampling = self.fs_layer(upsampling, self.ori_dim)
            upsampling = self.fs_layer(final_latent, self.ori_dim)
        return upsampling

    def CPGAN(self): 

        #tf.set_random_seed(9)
        self.data_p = tf.placeholder(tf.float32, shape=[None, self.ori_dim])
        self.label_p = tf.placeholder(tf.int64, shape=[None])
        self.one_hot = tf.one_hot(self.label_p, 2)
        self.noise_p = tf.placeholder(tf.float32, shape=[None, self.com_dim])
        self.emb_p = tf.placeholder(tf.float32, shape=[None, self.com_dim])


        #self.kernel_map = self.RFF_map(self.emb_p, self.seed, self.gamma, self.com_dim, self.ori_dim)

        with tf.variable_scope('white_box'):
            #white_box = self.fs_layer(self.emb_p, 20)
            #white_box = self.fs_layer(white_box, 24)
            #white_box = self.fs_layer(white_box, 28)
            #self.recon_white_box = self.fs_layer(white_box, self.ori_dim)
            self.recon_white_box = self.fs_layer(self.emb_p ,self.ori_dim)

        uti_update_white_box = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Decov")
        with tf.control_dependencies(uti_update_white_box):
            uti_update_white_box = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='white_box')
            self.theta_white_box = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='white_box')
            self.loss_white_box = tf.losses.mean_squared_error(self.data_p, self.recon_white_box)
            self.opt_white_box = tf.train.AdamOptimizer().minimize(self.loss_white_box, var_list=self.theta_white_box)


        with tf.variable_scope('privatizer'):
            #compressing = self.fs_layer(self.data_p, 20)
            #self.compressing = self.fs_layer(self.data_p, self.com_dim)
            self.compressing = self.fs_layer(self.data_p, 16)

            self.compressing = self.fs_layer(self.compressing, self.com_dim)
            #self.compressing = self.fs_layer(self.compressing, self.)

        self.theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='privatizer')

        with tf.variable_scope('classifier'):

            #pre_logit = self.fs_layer(self.compressing, 8)
            #pre_logit = self.fs_layer(pre_logit, 4)
            self.logit = self.fs_layer(self.compressing, 2)
            self.prob = tf.nn.softmax(self.logit)

        self.theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')
        #self.lrr_mu_p = tf.placeholder(tf.float32, shape=[self.com_dim])
        #self.krr_mu_p = tf.placeholder(tf.float32, shape=[self.mapping_dim])
        #self.t_mu_p = tf.placeholder(tf.float32, shape=[self.ori_dim])
        self.krr_weights_p = tf.placeholder(tf.float32, shape=[self.mapping_dim, self.ori_dim])
        self.lrr_weights_p = tf.placeholder(tf.float32, shape=[self.com_dim, self.ori_dim])

        #self.lrr_mu = self.init_tensor([self.com_dim])
        #self.krr_mu = self.init_tensor([self.mapping_dim])  
        #self.t_mu = self.init_tensor([self.ori_dim])

        self.recon_nn = self.adversary_nn(self.compressing)
        self.recon_lrr = self.adversary_lrr(self.compressing)
        self.kernel_map = self.RFF_map(self.compressing, self.seed, self.gamma, self.com_dim, self.mapping_dim)
        self.recon_krr = self.adversary_krr(self.kernel_map)

        ### Get weiths.
        self.theta_r_lrr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='adversary_lrr')
        self.theta_r_krr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='adversary_krr')
        self.theta_nn = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='adversary_nn')

        self.loss_r_lrr = tf.losses.mean_squared_error(self.data_p, self.recon_lrr) 
        self.loss_r_krr = tf.losses.mean_squared_error(self.data_p, self.recon_krr) 
        self.loss_r_nn = tf.losses.mean_squared_error(self.data_p, self.recon_nn) 

        self.loss_uti = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= self.one_hot, logits= self.logit))

        self.loss_g_lrr = self.trade_off * self.loss_uti - self.loss_r_lrr
        self.loss_g_krr = self.trade_off * self.loss_uti - self.loss_r_krr
        self.loss_g_nn  = self.trade_off * self.loss_uti - self.loss_r_nn

        self.assign_op = []
        assign_lrr = self.theta_r_lrr[0].assign(self.lrr_weights_p)
        self.assign_op.append(assign_lrr)

        assign_krr = self.theta_r_krr[0].assign(self.krr_weights_p)
        self.assign_op.append(assign_krr)

        #assign_t_mu = self.t_mu.assign(self.t_mu_p)
        #self.assign_op.append(assign_t_mu)

        #assign_lrr_mu = self.lrr_mu.assign(self.lrr_mu_p)
        #self.assign_op.append(assign_lrr_mu)

        #assign_krr_mu = self.krr_mu.assign(self.krr_mu_p)
        #self.assign_op.append(assign_krr_mu)

        uti_update = []
        scope = ['privatizer', 'classifier', 'adversary_lrr', 'adversary_nn', 'adversary_krr']
        for i in scope: 
            uti_update += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = i)

        with tf.control_dependencies(uti_update):

            self.c_op = tf.train.AdamOptimizer(0.001)
            self.c_opt = self.c_op.minimize(self.loss_uti, var_list=self.theta_c)

            self.r_op = tf.train.AdamOptimizer(0.001)
            self.r_opt = self.r_op.minimize(self.loss_r_nn, var_list=self.theta_nn)

            self.g_op_nn = tf.train.AdamOptimizer(0.001)
            self.g_opt_nn = self.g_op_nn.minimize(self.loss_g_nn, var_list=self.theta_g)

            '''
            self.g_op_lrr = tf.train.AdamOptimizer(0.001)
            self.g_opt_lrr = self.g_op_lrr.minimize(self.loss_g_lrr, var_list=self.theta_g)

            self.g_op_krr = tf.train.AdamOptimizer(0.001)
            self.g_opt_krr = self.g_op_krr.minimize(self.loss_g_krr, var_list=self.theta_g)
            '''

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=5)

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
        
        #print('Shape of KRR weights: {}.'.format(weights.shape))
        #clf = Ridge(alpha=0.001, fit_intercept=False, random_state=9)
        #clf.fit(emb_matrix, train_norm)
        #weights = clf.coef_
        return weights


    def LRR_close_form(self, emb_matrix, train_matrix):
        
        mu = np.mean(emb_matrix, axis=0)
        emb_matrix = emb_matrix - mu
        emb_matrix = emb_matrix.T
        rau = 0.001
        s = np.dot(emb_matrix, emb_matrix.T)
        h,w = s.shape
        s_inv = np.linalg.inv(s + rau*np.identity(h))
        train_mu  = np.mean(train_matrix, axis=0)
        train_norm = train_matrix - train_mu
        weights = np.dot(np.dot(s_inv, emb_matrix), train_norm)
        #weights = np.dot(np.dot(s_inv, emb_matrix), train_matrix)
        #clf = Ridge(alpha=0.001, fit_intercept=False, random_state=9)
        #clf.fit(emb_matrix, train_norm)
        #weights = clf.coef_
        #print('Shape of LRR weights: {}'.format(weights.shape))

        #rint(weights)
        return weights

    def get_emb_matrix(self, data): 

        count = 0

        temp_lrr = []
        temp_krr = []

        for i,j in self.next_batch(data, self.t_label, self.batch_size):

            uu, yy = self.sess.run([self.compressing, self.kernel_map], feed_dict={self.data_p:i})
            temp_lrr.append(uu)
            temp_krr.append(yy)

        #emb_matrix_lrr = np.concatenate((emb_matrix_lrr, uu), axis=0)
        #emb_matrix_krr = np.concatenate((emb_matrix_krr, yy), axis=0)

        emb_matrix_lrr = np.concatenate(temp_lrr, axis=0)
        emb_matrix_krr = np.concatenate(temp_krr, axis=0)

        return emb_matrix_lrr, emb_matrix_krr 


    def get_train_matrix(self): 
        print('Successfully get flatted train matrix !!!!')     
        return self.t_data


    def assign(self, train_matrix):

        ### Multiple adversaries CPGAN only !!!!

        feed_dict_assign = {}

        emb_matrix_lrr, emb_matrix_krr = self.get_emb_matrix(self.t_data)

        error_list = []
        update_choice = [self.g_opt_nn, self.g_opt_lrr, self.g_opt_krr]
        #assign_op = []

        lrr_weights = self.LRR_close_form(emb_matrix_lrr, train_matrix)
        #feed_dict_assign[self.lrr_mu_p] = lrr_mu
        feed_dict_assign[self.lrr_weights_p] = lrr_weights

        #assign_op.append(tf.assign(self.theta_r_lrr[0], lrr_weights))

        krr_weights = self.KRR_close_form(emb_matrix_krr, train_matrix)
        #feed_dict_assign[self.krr_mu_p] = krr_mu
        feed_dict_assign[self.krr_weights_p] = krr_weights
        #feed_dict_assign[self.t_mu_p] = train_mu
        #assign_op.append(tf.assign(self.theta_r_krr[0], krr_weights))

        self.sess.run(self.assign_op, feed_dict = feed_dict_assign)

        error_nn, error_lrr, error_krr = self.compute_reco_mse(self.v_data, self.v_label)
        error_list.append(error_nn) 
        error_list.append(error_lrr)
        error_list.append(error_krr)
        print('Average MSE among all testing images is {}, {}, {}.(nn,lrr,krr)'.format(error_nn, error_lrr, error_krr))
        optimize_g = update_choice[np.argmin(error_list)]

        return optimize_g, feed_dict_assign


    def compute_reco_mse(self, data, label):

        ##### after assign all the weights !!!!! 

        error_nn = []
        error_lrr = []
        error_krr = []

        for i , j in self.next_batch(data, label, self.batch_size):

            b = i.shape[0]
            no = np.random.normal(size=(b, self.ori_dim))
            #no = np.random.laplace(size=(b, 175, 175,3))
            feed_dict = {} 
            feed_dict[self.data_p] = i
            up_nn = self.sess.run(self.recon_nn, feed_dict=feed_dict)
            up_lrr = self.sess.run(self.recon_lrr, feed_dict=feed_dict)
            up_krr = self.sess.run(self.recon_krr, feed_dict=feed_dict)

            for k in range(len(up_nn)):
                #error.append(mean_squared_error(self.plot(i[k]).flatten(),self.plot(up[k]).flatten()))
                error_nn.append(mean_squared_error(i[k], up_nn[k]))
                error_lrr.append(mean_squared_error(i[k], up_lrr[k]))
                error_krr.append(mean_squared_error(i[k], up_krr[k]))

        return np.mean(error_nn), np.mean(error_lrr), np.mean(error_krr)


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

    def train(self):

        train_matrix = self.get_train_matrix()
        epo = 1

        for _ in range(self.epo):
            print("Epoch {}.".format(epo))
            #citers = 25
            temp_t = []
            temp_l = []

            for i, j in self.next_batch(self.t_data, self.t_label, self.batch_size, shuffle=True):

                sample = i.shape[0]
                #no  = np.random.normal(size=(sample, self.com_dim))
                feed_dict = {} 
                feed_dict[self.data_p] = i.reshape(-1, self.ori_dim)
                feed_dict[self.label_p] = j.reshape(-1)

                for _ in range(self.citer):
                    privacy_loss, _ = self.sess.run([self.loss_r_nn, self.r_opt], feed_dict=feed_dict)
                u_loss, _ = self.sess.run([self.loss_uti, self.c_opt], feed_dict=feed_dict)
                _ = self.sess.run(self.g_opt_nn, feed_dict=feed_dict)
                #temp_t.append(i)
                #temp_l.append(j)

            #optimize_g, feed_dict = self.assign(train_matrix)

            #for k in range(len(temp_t)):

                #feed_dict[self.data_p] = temp_t[k]
                #feed_dict[self.label_p] = temp_l[k]
                #_ = self.sess.run([optimize_g], feed_dict=feed_dict)
            epo +=1 
        acc = self.prediction_and_accuracy(self.v_data, self.v_label)
        mse, mse_lrr, mse_krr = self.evalute_privacy(self.t_data, self.v_data, self.v_label)

        theta_list = self.sess.run(self.theta_g)
        weights = np.dot(theta_list[0], theta_list[1])
        #weights = self.sess.run(self.theta_g)[0]
        #theory_acc = self.compute_theory_acc(self.sess.run(self.theta_g)[0].T)
        #theory_mse = self.compute_theory_mse(self.sess.run(self.theta_g)[0].T)
        theory_acc = self.compute_theory_acc(weights.T)
        theory_mse = self.compute_theory_mse(weights.T)

        return acc, mse, theory_acc, theory_mse, mse_lrr, mse_krr

    def prediction_and_accuracy(self, data, label):

        pred = [] 
        for i, j in self.next_batch(data, label, self.batch_size):

            sample = i.shape[0]
            #no = np.random.multivariate_normal([0 for i in range(self.com_dim)], np.identity(self.com_dim), sample)
            no  = np.random.laplace(size=(sample, self.com_dim))
            feed_dict = {} 
            feed_dict[self.data_p] = i
            #feed_dict[self.noise_p] = self.factor * no
            prob = self.sess.run(self.prob, feed_dict=feed_dict)
            pred.append(prob)

        predict = np.concatenate(pred, axis=0)
        predict = np.argmax(predict, axis=1)
        accuracy = accuracy_score(predict, np.array(self.v_label))       
        print("Utility Accuracy: {}".format(accuracy))
        return accuracy

    def evalute_privacy(self, data, t_data, label):

        data_list = []
        emb_list = []
        map_list = []
        epo = 1 

        for _ in range(100):
            for i, j in self.next_batch(data, label, self.batch_size):
                sample = i.shape[0]
                emb, RFF_map = self.sess.run([self.compressing, self.kernel_map], feed_dict = {self.data_p:i})

                _ = self.sess.run(self.opt_white_box, feed_dict={self.emb_p:emb, self.data_p:i})
                if epo == 1 : 
                    data_list.append(i)
                    emb_list.append(emb)
                    map_list.append(RFF_map)
            epo += 1 

        feed_dict={}
        feed_dict[self.krr_weights_p] = self.KRR_close_form(np.concatenate(map_list,axis=0), self.t_data)
        feed_dict[self.lrr_weights_p] = self.LRR_close_form(np.concatenate(emb_list,axis=0), self.t_data)
        self.sess.run(self.assign_op, feed_dict=feed_dict)

        #emb_data = 
        mse_list = []
        mse_list_lrr = []
        mse_list_krr = []
        for i, j in self.next_batch(t_data, label, self.batch_size):

            feed_dict = {}
            feed_dict[self.data_p] = i
            feed_dict[self.emb_p] = self.sess.run(self.compressing, feed_dict={self.data_p:i})
            reco, reco_lrr, recon_krr = self.sess.run([self.recon_white_box, self.recon_lrr, self.recon_krr], feed_dict=feed_dict)

            for k in range(len(reco)):
                mse_list.append(mean_squared_error(i[k], reco[k])*self.ori_dim)
                mse_list_lrr.append(mean_squared_error(i[k], reco_lrr[k])*self.ori_dim)
                mse_list_krr.append(mean_squared_error(i[k], recon_krr[k])*self.ori_dim)

        mean_mse = np.mean(mse_list)
        mean_mse_lrr = np.mean(mse_list_lrr)
        mean_mse_krr = np.mean(mse_list_krr)

        print("white_box attack MSE (NN): {}.".format(mean_mse))
        print("white_box attack MSE (LRR): {}.".format(mean_mse_lrr))
        print("white_box attack MSE (KRR): {}.".format(mean_mse_krr))

        return mean_mse, mean_mse, mean_mse_krr


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
        value = np.trace(self.cov_x - temp) #/ self.ori_dim
        print("Theoretical mse: {}".format(value))

        return value


    def std_normal(self, x):
        return 1/sqrt(2*pi) * exp(-(x**2)/2)

    def Qfunc_tail(self, y):
        I = quad(self.std_normal, y,  np.inf)
        return I 


