import numpy as np 
import pandas as pd
import random 
import time 
import math
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
from sklearn.utils import check_random_state
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA ## For resize. 
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from sklearn.kernel_ridge import KernelRidge
np.random.seed(9)
random.seed(9)
#tf.set_random_seed(9)

class DNN:
    def __init__(self, arg):

        #np.random.seed(9)
        #random.seed(9)
        #tf.set_random_seed(9)

        self.arg = arg 
        self.ori_dim = self.arg.ori_dim
        self.com_dim = self.arg.com_dim
        self.samples = self.arg.samples
        self.noise_scale = self.arg.noise_scale
        self.noise_factor = self.arg.noise_term
        self.batch_size = self.arg.batch_size
        self.seed = int(self.arg.seed)
        self.gamma = self.arg.gamma
        self.mapping_dim = self.arg.mapping_dim
        self.mapping_dim_pca = self.arg.mapping_dim_pca
        self.epo = self.arg.epoch
        self.pca_dim = self.arg.pca_dim
        self.mu = np.array([[2 for i in range(self.ori_dim)]]).reshape(self.ori_dim, 1)
        self.prior_prob = self.arg.prior_prob
        self.t_data, self.t_label, self.v_data, self.v_label, self.cov_x, self.cov_s = self.generate_data(self.samples, self.prior_prob, self.mu)
        self.DNN_with_resize()

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


    def adversary_lrr(self, final_latent, name, reuse=False):
        # If deduct mu, then.....
        with tf.variable_scope('adversary_lrr_'+name) as scope:  
            if reuse: 
                scope.reuse_variables()
            recontruction = ly.fully_connected(final_latent, self.ori_dim, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = None)
        #return tf.reshape(recontruction, shape=[-1, 175, 175, 3])
        return recontruction


    def adversary_krr(self, kernel_map, name, reuse=False):
        # If deduct mu, then.....
        with tf.variable_scope('adversary_krr_'+name) as scope:  
            if reuse: 
                scope.reuse_variables()
            recontruction = ly.fully_connected(kernel_map, self.ori_dim, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = None)
        return recontruction


    def adversary_nn(self, final_latent, reuse=False):

        with tf.variable_scope('adversary_nn_'+name) as scope:
            if reuse:
                scope.reuse_variables()
            upsampling = self.fs_layer(final_latent, 20)
            upsampling = self.fs_layer(upsampling, 24)
            upsampling = self.fs_layer(upsampling, 28)
            upsampling = self.fs_layer(upsampling, self.ori_dim)
        return upsampling

    def RFF_map(self, input_tensor, seed, stddev, input_shape, output_dim):
        #input_tensor = tf.concat([input_tensor_1, input_tensor_2], axis=1)
        #print("Information that the adversary can get: {}".format(input_tensor))

        #random_state = check_random_state(seed)
        gamma = stddev
        omega_matrix_shape = [input_shape, output_dim]
        bias_shape = [output_dim]
        
        #This is the tensorflow version RFF mapping, but I refer to the scikit-learn version.!!!!
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #self._stddev = stddev
        '''
        omega_matrix = constant_op.constant(
            np.random.normal(
            scale=1.0 / self._stddev, size=omega_matrix_shape),
            dtype=dtypes.float32)
        
        omega_matrix = constant_op.constant(
            np.random.normal(
            scale=1.0 / gamma, size=omega_matrix_shape),
            dtype=dtypes.float32)

        bias = constant_op.constant(
            np.random.uniform(
            low=0.0, high=2 * np.pi, size=bias_shape),
            dtype=dtypes.float32)

        x_omega_plus_bias = math_ops.add(
            math_ops.matmul(input_tensor, omega_matrix), bias)

        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        
        omega_matrix = constant_op.constant(np.sqrt(2 * gamma) *
           random_state.normal(size=omega_matrix_shape), dtype=dtypes.float32)

        bias = constant_op.constant(
            random_state.uniform(
            0.0, 2 * np.pi, size=bias_shape), dtype=dtypes.float32)

        x_omega_plus_bias = math_ops.add(
            math_ops.matmul(input_tensor, omega_matrix), bias)
        '''
        omega_matrix = constant_op.constant(np.sqrt(2 * gamma) *
           np.random.normal(size=omega_matrix_shape), dtype=dtypes.float32)

        bias = constant_op.constant(
            np.random.uniform(
            0.0, 2 * np.pi, size=bias_shape), dtype=dtypes.float32)

        x_omega_plus_bias = math_ops.add(
            math_ops.matmul(input_tensor, omega_matrix), bias)

        return math.sqrt(2.0 / output_dim) * math_ops.cos(x_omega_plus_bias)

    def DNN_with_resize(self):

        self.data_p = tf.placeholder(tf.float32, shape=[None, self.ori_dim])
        self.label_p = tf.placeholder(tf.int64, shape=[None])
        self.one_hot = tf.one_hot(self.label_p, 2)

        self.com_input_p = tf.placeholder(tf.float32, shape=[None, self.com_dim])

        self.pca_input = tf.placeholder(tf.float32, shape=[None, self.pca_dim])
        self.pca_noise = tf.placeholder(tf.float32, shape=[None ,self.pca_dim])
        self.perturbated_data = tf.add(self.pca_input, self.noise_factor * self.pca_noise)

        self.com_map = self.RFF_map(self.com_input_p, self.seed, self.gamma, self.com_dim, self.mapping_dim)
        self.pca_map = self.RFF_map(self.perturbated_data, self.seed, self.gamma, self.pca_dim, self.mapping_dim_pca)

        self.krr_weights_white_box_p = tf.placeholder(tf.float32, shape=[self.mapping_dim, self.ori_dim])
        self.lrr_weights_white_box_p = tf.placeholder(tf.float32, shape=[self.com_dim, self.ori_dim])
        
        self.krr_weights_pca_p = tf.placeholder(tf.float32, shape=[self.mapping_dim_pca, self.ori_dim])
        self.lrr_weights_pca_p = tf.placeholder(tf.float32, shape=[self.pca_dim, self.ori_dim])

        self.upsampling_white_box_krr = self.adversary_krr(self.com_map, "white_box")
        self.upsampling_white_box_lrr = self.adversary_lrr(self.com_input_p, "white_box")


        self.upsampling_pca_krr = self.adversary_krr(self.pca_map, "pca")
        self.upsampling_pca_lrr = self.adversary_lrr(self.perturbated_data, "pca")

        self.theta_krr_white_box = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='adversary_krr_white_box')
        self.theta_lrr_white_box = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='adversary_lrr_white_box')

        self.theta_krr_pca = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='adversary_krr_pca')
        self.theta_lrr_pca = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='adversary_lrr_pca')

        self.assign_op = []
        self.assign_op.append(self.theta_krr_white_box[0].assign(self.krr_weights_white_box_p))
        self.assign_op.append(self.theta_lrr_white_box[0].assign(self.lrr_weights_white_box_p))
        self.assign_op.append(self.theta_krr_pca[0].assign(self.krr_weights_pca_p))
        self.assign_op.append(self.theta_lrr_pca[0].assign(self.lrr_weights_pca_p))


        with tf.variable_scope("Deep"):

            compressing = self.fs_layer(self.data_p, 24)
            self.compressing = self.fs_layer(compressing, self.com_dim)

        with tf.variable_scope("Classifier"):
            '''
            pre_logit = self.fs_layer(self.compressing, 8)
            pre_logit = self.fs_layer(pre_logit, 4)
            self.logit = self.fs_layer(pre_logit, 2)    
            self.prob = tf.nn.softmax(self.logit)
            '''

            self.logit = self.fs_layer(self.compressing, 2)
            self.prob = tf.nn.softmax(self.logit)

        with tf.variable_scope("Decov"):

            #upsampling = self.fs_layer(self.com_input_p, 20)
            #upsampling = self.fs_layer(upsampling, 24)
            #upsampling = self.fs_layer(upsampling, 28)
            self.upsampling = self.fs_layer(self.com_input_p, self.ori_dim)         


        with tf.variable_scope("Classifier_PCA"):
            '''
            pre_logit_pca = self.fs_layer(self.perturbated_data, 12)
            pre_logit_pca = self.fs_layer(pre_logit_pca, 8)
            self.logit_pca = self.fs_layer(pre_logit_pca, 2)  
            self.prob_pca = tf.nn.softmax(self.logit_pca)
            '''
            self.logit_pca = self.fs_layer(self.pca_input)
            self.prob_pca = tf.nn.softmax(self.logit_pca)
        with tf.variable_scope("Decoder_PCA"):

            #upsampling_pca = self.fs_layer(self.perturbated_data, 20)
            #upsampling_pca = self.fs_layer(upsampling_pca, 24)
            #upsampling_pca = self.fs_layer(upsampling_pca, 28)
            self.upsampling_pca = self.fs_layer(self.pca_input, self.ori_dim)  


        ## while-box reconstruction attack, excluding the model inversion stage.
        self.theta_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Deep')
        self.theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Classifier')
        self.theta_r = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decov')

        self.loss_c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= self.one_hot, logits=self.logit))
        self.loss_r = tf.losses.mean_squared_error(self.data_p, self.upsampling)

        uti_update = []
        scope = ["Deep", "Classifier"]
        for i in scope: 
            uti_update += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=i)
        with tf.control_dependencies(uti_update):
            op = tf.train.AdamOptimizer(0.001)
            self.opt = op.minimize(self.loss_c, var_list = self.theta_d + self.theta_c)


        uti_update_decov = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Decov")
        with tf.control_dependencies(uti_update_decov):
            op_r = tf.train.AdamOptimizer(0.001)
            self.opt_r = op_r.minimize(self.loss_r, var_list = self.theta_r)

        ### FOR PCA LEARNING NETWORKS.

        self.theta_c_pca = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Classifier_PCA')
        self.theta_d_pca = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decoder_PCA')

        self.loss_c_pca = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= self.one_hot, logits=self.logit_pca))
        self.loss_r_pca = tf.losses.mean_squared_error(self.data_p, self.upsampling_pca)


        uti_update_pca_d = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Decoder_PCA")
        with tf.control_dependencies(uti_update_pca_d):
            op_r_pca = tf.train.AdamOptimizer(0.001)
            self.opt_r_pca = op_r_pca.minimize(self.loss_r_pca, var_list = self.theta_d_pca)


        uti_update_pca_c = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Classifier_PCA")
        with tf.control_dependencies(uti_update_pca_c):
            op_c_pca = tf.train.AdamOptimizer(0.001)
            self.opt_c_pca = op_c_pca.minimize(self.loss_c_pca, var_list = self.theta_c_pca)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        ## First end-to-end train the deep model while computing the accuracy.

    def inject_noise(self, t_emb_pca, v_emb_pca):

        t_emb_pca = t_emb_pca + self.noise_factor * np.random.laplace(0, self.noise_scale, size=(t_emb_pca.shape[0], self.pca_dim))
        v_emb_pca = v_emb_pca + self.noise_factor * np.random.laplace(0, self.noise_scale, size=(t_emb_pca.shape[0], self.pca_dim))

        return t_emb_pca, v_emb_pca

    def train(self):

        for _ in range(self.epo): 
            for i, j in self.next_batch(self.t_data, self.t_label, self.batch_size, shuffle=True):
                sample = i.shape[0]
                #no  = np.random.laplace(0, 0.45, size=(sample, self.com_dim))
                no = np.random.laplace(0, 0.5, size=(sample, self.com_dim))
                feed_dict = {} 
                feed_dict[self.data_p] = i.reshape(-1, self.ori_dim)
                feed_dict[self.label_p] = j.reshape(-1)
                _ = self.sess.run(self.opt, feed_dict=feed_dict)

        ## Second, traing the devoc networks for the deep feature 
        ## while computing the MSE.

        t_emb = self.get_emb(self.t_data)
        #print(t_emb[0])
        v_emb = self.get_emb(self.v_data)

        pca = PCA(n_components = self.pca_dim, random_state=9)

        t_emb_pca = pca.fit_transform(t_emb)
        #print(t_emb_pca)
        v_emb_pca = pca.transform(v_emb)
        #print(v_emb_pca)

        t_emb_pca, v_emb_pca = self.inject_noise(t_emb_pca, v_emb_pca)

        print("************* Compute all weights *************")
        feed_dict[self.krr_weights_white_box_p] = self.KRR_close_form(t_emb, self.t_data, map=True)
        print("************* Compute all weights *************")

        feed_dict[self.lrr_weights_white_box_p] = self.LRR_close_form(t_emb, self.t_data)
        print("************* Compute all weights *************")

        feed_dict[self.krr_weights_pca_p] = self.KRR_close_form(t_emb_pca, self.t_data, map=True, pca=True)
        print("************* Compute all weights *************")

        feed_dict[self.lrr_weights_pca_p] = self.LRR_close_form(t_emb_pca, self.t_data, pca=True)
        print("************* Compute all weights *************")

        self.sess.run(self.assign_op, feed_dict=feed_dict)
        ### may be incorporated in one function.


        print("********** Evaluation of the deep features **********")
        for _ in range(100):

            for i, j, k  in self.eva_next_batch(t_emb, self.t_data, self.t_label, self.batch_size, shuffle=False):
                sample = i.shape[0]
                #no = np.random.multivariate_normal([0 for i in range(self.com_dim)], np.identity(self.com_dim), sample)
                #no  = np.random.laplace(0, 0.45, size=(sample, self.com_dim))
                no = np.random.laplace(0, self.noise_factor, size=(sample, self.com_dim))
                feed_dict = {} 
                #aa = self.sess.run(self.compressing, feed_dict={self.data_p:j})
                feed_dict[self.com_input_p] = i
                #print(aa[0])
                #feed_dict[self.compressing] = i.reshape(-1, self.com_dim)
                feed_dict[self.data_p] = j
                _ = self.sess.run(self.opt_r, feed_dict=feed_dict)

        mse, mse_lrr, mse_krr = self.prediction_and_mse(self.v_data, self.v_label, PCA=False)
        acc = self.prediction_and_accuracy(v_emb, self.v_label, PCA=False)

        print("********** Evaluation of the PCA features **********")

        for _ in range(100):
            for i, j, k in self.eva_next_batch(t_emb_pca, self.t_data, self.t_label, self.batch_size, shuffle=False):
                sample = i.shape[0]
                #no  = np.random.laplace(0, 0.45, size=(sample, self.pca_dim))
                no = np.random.laplace(0, self.noise_scale, size=(sample, self.pca_dim))
                feed_dict = {} 
                feed_dict[self.pca_input] = i
                feed_dict[self.data_p] = j
                feed_dict[self.pca_noise] = no 
                feed_dict[self.label_p] = k

                _, _ = self.sess.run([self.opt_r_pca, self.opt_c_pca], feed_dict=feed_dict)

        mse_pca, mse_pca_lrr, mse_pca_krr = self.prediction_and_mse(v_emb_pca, self.v_label, PCA=True)
        acc_pca = self.prediction_and_accuracy(v_emb_pca, self.v_label, PCA=True)

        return acc, mse, mse_lrr, mse_krr, acc_pca, mse_pca, mse_pca_lrr, mse_pca_krr

    def get_emb(self, data):
        
        temp = []
        for i, j in self.next_batch(data, self.t_label, self.batch_size, shuffle=False):

                feed_dict = {} 
                feed_dict[self.data_p] = i.reshape(-1, self.ori_dim)
                #feed_dict[self.label_p] = j.reshape(-1)      
                temp.append(self.sess.run(self.compressing, feed_dict=feed_dict))

        return np.concatenate(temp, axis=0)

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


    def eva_next_batch(self, t_emb, t_data, t_label, batch_size, shuffle=False):

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
                yield np.array(t_emb[i:]), np.array(t_data[i:]) , np.array(t_label[i:])
            else : 
                yield np.array(t_emb[i: i+self.batch_size]), np.array(t_data[i: i+self.batch_size]) , np.array(t_label[i: i+self.batch_size])



    def prediction_and_accuracy(self, data, label, PCA=True):

        pred = [] 

        for i, j in self.next_batch(data, label, self.batch_size):

            sample = i.shape[0]
            feed_dict = {} 

            if PCA: 
                #no  = np.random.laplace(0, 0.45, size=(sample, self.pca_dim))
                no = np.random.laplace(0, self.noise_scale, size=(sample, self.pca_dim))
                feed_dict[self.pca_input] = i
                feed_dict[self.pca_noise] = no 
                prob = self.sess.run(self.prob_pca, feed_dict=feed_dict)

            else : 
                #feed_dict[self.com_input_p] = i
                feed_dict[self.compressing] = i
                prob = self.sess.run(self.prob, feed_dict=feed_dict)

            pred.append(prob)

        predict = np.concatenate(pred, axis=0)
        predict = np.argmax(predict, axis=1)

        TP = [] 
        NP = []
        for i, j in zip(self.v_label, predict):
            #print(i)
            if i == 1 : 
                if i == j : 
                    TP.append(1)
            if i == 0 : 
                if i == j : 
                    NP.append(1)

        acc = (len(TP) + len(NP)) / len(self.v_data)
        accuracy = accuracy_score(predict, np.array(self.v_label))

        if PCA :
            print("PCA with noise Accuracy: {}".format(accuracy))      
        else: 
            print("Deep features Accuracy: {}".format(accuracy))
        return accuracy

    def prediction_and_mse(self, data, label, PCA=True): 

        pred = []
        pred_lrr = []
        pred_krr = []
        for i, j in self.next_batch(data, label, self.batch_size):

            sample = i.shape[0]
            #no = np.random.multivariate_normal([0 for i in range(self.com_dim)], np.identity(self.com_dim), sample)
            feed_dict = {} 
            if PCA : 
                #no  = np.random.laplace(0, 0.45, size=(sample, self.pca_dim))
                no  = np.random.laplace(0, self.noise_scale, size=(sample, self.pca_dim))

                feed_dict[self.pca_input] = i
                feed_dict[self.pca_noise] = no
                reco, reco_lrr, reco_krr = self.sess.run([self.upsampling_pca, self.upsampling_pca_lrr, self.upsampling_pca_krr], feed_dict=feed_dict)
            else :
                feed_dict[self.data_p] = i 
                kk = self.sess.run(self.compressing, feed_dict=feed_dict)
                feed_dict[self.com_input_p] = kk
                #feed_dict[self.compressing] = i
                reco, reco_lrr, reco_krr = self.sess.run([self.upsampling, self.upsampling_white_box_lrr, self.upsampling_white_box_krr], feed_dict=feed_dict)
            pred.append(reco)
            pred_lrr.append(reco_lrr)
            pred_krr.append(reco_krr)

        predict = np.concatenate(pred, axis=0)
        predict_lrr = np.concatenate(pred_lrr, axis=0)
        predict_krr = np.concatenate(pred_krr, axis=0)

        error = [] 
        error_lrr = []
        error_krr = []

        for i in range(len(predict)):
            error.append(mean_squared_error(self.v_data[i], predict[i])*self.ori_dim)
            error_lrr.append(mean_squared_error(self.v_data[i], predict_lrr[i])*self.ori_dim)
            error_krr.append(mean_squared_error(self.v_data[i], predict_krr[i])*self.ori_dim)

        
        mse = np.mean(error) 
        mse_lrr = np.mean(error_lrr)
        mse_krr = np.mean(error_krr)
        '''
        mse = np.sum(error) 
        mse_lrr = np.sum(error_lrr)
        mse_krr = np.sum(error_krr)
        '''
        if PCA: 
            print("PCA with noise MSE (NN): {}".format(mse))
            print("PCA with noise MSE (LRR): {}".format(mse_lrr))
            print("PCA with noise MSE (KRR): {}".format(mse_krr))
        else : 
            print("Deep features MSE (NN): {}".format(mse))
            print("Deep features MSE (LRR): {}".format(mse_lrr))
            print("Deep features MSE (KRR): {}".format(mse_krr))

        return mse, mse_lrr, mse_krr

    def KRR_close_form(self, emb, train_matrix, map=False, pca=False):
        # Use the random fourier transform to approximate the RBF kernel 
        # Note that the training data is too large so that we use the intrinsic space mapping 
        # And use the tensorflow conrtrib package to get the RFF mapping rather than hand crafting
        # More information refers to https://github.com/hichamjanati/srf  
        #train_matrix+=1 
        #train_matrix *= 127.5
        temp = []

        if pca : 
            #emb_matrix = emb
            #a, b = emb.shape[0], emb.shape[1]
            #no = np.random.laplace(size=(a, b))
            if map : 
                for i, j in self.next_batch(emb, self.t_label, self.batch_size):
                    feed_dict = {}
                    sample = i.shape[0]
                    feed_dict[self.pca_input] = i
                    #feed_dict[self.pca_noise] =  np.random.laplace(0, 0.45, size=(sample, self.pca_dim))
                    feed_dict[self.pca_noise] =  np.random.laplace(0, self.noise_scale, size=(sample, self.pca_dim))

                    temp.append(self.sess.run(self.pca_map, feed_dict=feed_dict))
                emb_matrix = np.concatenate(temp, axis=0)

            else : 

                a,b = emb.shape
                #no = np.random.laplace(0, 0.45, size=(a, b))
                no = np.random.laplace(0, self.noise_scale, size=(a, b))
                #emb_matrix = emb + self.noise_factor * no
                emb_matrix = emb
                #emb_matrix = emb_matrix + (self.noise_factor * no)
        else : 

            if map: 
                for i, j in self.next_batch(emb, self.t_label, self.batch_size):
                    feed_dict = {}
                    sample = i.shape[0]
                    feed_dict[self.com_input_p] = i
                    temp.append(self.sess.run(self.com_map, feed_dict=feed_dict))
                emb_matrix = np.concatenate(temp, axis=0)

            else : 
                emb_matrix = emb

        rau = 1
        mu = np.mean(emb_matrix, axis=0)
        emb_matrix = emb_matrix - mu
        
        emb_matrix = emb_matrix.T 
        #print(np.mean(emb_matrix))
        s = np.dot(emb_matrix, emb_matrix.T)
        a,b = s.shape
        identity = np.identity(a)
        s_inv = np.linalg.inv(s + rau * np.identity(a))
        train_mu = np.mean(train_matrix, axis=0)
        train_norm = train_matrix - train_mu
        #weights = np.dot(np.dot(s_inv,emb_matrix), train_norm)
        weights = np.dot(np.dot(s_inv, emb_matrix), train_norm)

        train_mu = np.mean(train_matrix, axis=0)
        train_norm = train_matrix - train_mu
        #clf = Ridge(alpha=1, fit_intercept=False, random_state=9)
        #clf = KernelRidge(alpha=0.01, kernel='rbf')
        #clf.fit(emb_matrix, train_norm)
        #clf.fit(emb_matrix, train_matrix)
        #weights = clf.coef_
        print('Shape of KRR weights: {}.'.format(weights.shape))
        return weights
    

    def LRR_close_form(self, emb, train_matrix, map=False, pca=False):

        #train_matrix+=1 
        #train_matrix *= 127.5
        temp = []
        if pca : 
            #emb_matrix = emb
            #a, b = emb.shape[0], emb.shape[1]
            #no = np.random.laplace(size=(a, b))
            if map : 
                for i, j in self.next_batch(emb, self.t_label, self.batch_size):
                    feed_dict = {}
                    sample = i.shape[0]
                    feed_dict[self.pca_input] = i
                    #feed_dict[self.pca_noise] = np.random.laplace(0, 0.45, size=(sample, self.pca_dim))
                    feed_dict[self.pca_noise] = np.random.laplace(0, self.noise_scale, size=(sample, self.pca_dim))

                    temp.append(self.sess.run(self.pca_map, feed_dict=feed_dict))
                emb_matrix = np.concatenate(temp, axis=0)
            else : 
                a, b = emb.shape
                #no = np.random.laplace(0, 0.45, size=(a, b))
                no = np.random.laplace(0, self.noise_scale, size=(a, b))
                #emb_matrix = emb + self.noise_factor* no
                emb_matrix = emb
                #emb_matrix = emb_matrix + (self.noise_factor * no)
        else : 
            if map : 
                for i, j in self.next_batch(emb, self.t_label, self.batch_size):
                    feed_dict = {}
                    sample = i.shape[0]
                    feed_dict[self.com_input_p] = i
                    temp.append(self.sess.run(self.com_map, feed_dict=feed_dict))
                emb_matrix = np.concatenate(temp, axis=0)
            else : 
                emb_matrix = emb

        mu = np.mean(emb_matrix, axis=0)
        emb_matrix = emb_matrix - mu
        
        emb_matrix = emb_matrix.T
        rau = 0.001
        s = np.dot(emb_matrix, emb_matrix.T)
        h,w = s.shape
        s_inv = np.linalg.inv(s + rau * np.identity(h))        
        train_mu = np.mean(train_matrix, axis=0)
        train_norm = train_matrix - train_mu
        #clf = Ridge(alpha=1, fit_intercept=False, random_state=9)
        #clf.fit(emb_matrix, train_norm)
        #clf.fit(emb_matrix, train_matrix)
        #weights = clf.coef_ 
        weights = np.dot(np.dot(s_inv, emb_matrix), train_norm)
        #weights = np.dot(np.dot(s_inv, emb_matrix), train_matrix)
        print('Shape of LRR weights: {}'.format(weights.shape))
        return weights


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








