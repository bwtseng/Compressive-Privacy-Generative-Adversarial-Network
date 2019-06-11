from har import load_data
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

#np.random.seed(9)
#random.seed(9)
class DNN:

    def __init__(self, arg):
        
        self.arg = arg 
        self.com_dim = self.arg.dim
        self.ori_dim = 128 * 1 * 9
        self.noise_scale = self.arg.noise_scale
        self.noise_factor = self.arg.noise_term
        self.epo = self.arg.epoch
        self.mapping_dim = self.arg.mapping_dim
        self.mapping_dim_pca = self.arg.mapping_dim_pca
        self.gamma = self.arg.gamma
        self.seed = self.arg.seed
        self.pca_dim = self.arg.pca_dim
        self.batch_size = self.arg.batch_size
        self.t_data, self.t_label, self.v_data, self.v_label = load_data()
        self.DNN_with_Resize()

    def deconv(self, x, num_outputs, kernel_size, stride, activetion, pad): 
        return ly.conv2d_transpose(x, num_outputs, kernel_size=kernel_size, stride=(stride, 1), activation_fn=activetion, padding=pad, weights_initializer=tf.contrib.layers.xavier_initializer())

    def conv(self, x, kernel_size, stride, activetion, pad): 
        return ly.conv2d(x, kernel_size=kernel_size, stride=(stride, 1), activation_fn=activetion, padding=pad, weights_initializer=tf.contrib.layers.xavier_initializer())

    def fs_layer(self, x, units, activation): 
        return ly.fully_connected(x, units, activation_fn=activation, weights_initializer=tf.contrib.layers.xavier_initializer())

    def adversary_krr(self, kernel_vec, name, reuse=False):
        #final_latent = tf.concat([latent, latent_1], axis=1)
        with tf.variable_scope('adversary_krr_'+name) as scope:  
            if reuse: 
                scope.reuse_variables()
            recontruction = ly.fully_connected(kernel_vec, self.ori_dim, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = None)
        #return tf.reshape(recontruction, shape=[-1, 175, 175, 3])
        return tf.reshape(recontruction, shape=[-1, 128, 1, 9])


    def adversary_lrr(self, compressive_data, name, reuse=False):
        #final_latent = tf.concat([latent, latent_1], axis=1)
        with tf.variable_scope('adversary_lrr_'+name) as scope:  
            if reuse: 
                scope.reuse_variables()
            recontruction = ly.fully_connected(compressive_data, self.ori_dim, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = None)
        #return tf.reshape(recontruction, shape=[-1, 175, 175, 3])
        return tf.reshape(recontruction, shape=[-1, 128, 1, 9])

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

    def DNN_with_Resize(self):

        self.data_p = tf.placeholder(tf.float32, shape=[None, 128, 1, 9])
        self.label_p = tf.placeholder(tf.int64, shape=[None, 6])
        self.com_input_p = tf.placeholder(tf.float32, shape=[None, self.com_dim])
        self.pca_input = tf.placeholder(tf.float32, shape=[None, self.pca_dim])
        self.pca_noise = tf.placeholder(tf.float32, shape=[None, self.pca_dim])


        #self.compressing, self.logit = self.Alex_net(self.data_p, "Deep", "Classifier")
        self.compressing, self.logit = self.LeNet(self.data_p, "Deep", "Classifier")
        self.prob = tf.nn.softmax(self.logit)
        self.perturbated_data = tf.add(self.pca_input, self.noise_factor * self.pca_noise)
        self.com_map = self.RFF_map(self.com_input_p, self.seed, self.gamma, self.com_dim, self.mapping_dim)
        self.pca_map = self.RFF_map(self.perturbated_data, self.seed, self.gamma, self.pca_dim, self.mapping_dim_pca)


        with tf.variable_scope('Decov'):

            reshape = tf.reshape(self.com_input_p, shape=[-1, 4, 1, 128])
            deconv_white_box = self.deconv(reshape, 64, 3, 4, tf.nn.relu, 'SAME')
            deconv_white_box = self.deconv(deconv_white_box, 32, 3, 2, tf.nn.relu, 'SAME')
            deconv_white_box = self.deconv(deconv_white_box, 16, 3, 2, tf.nn.relu, 'SAME')
            self.deconv_white_box = self.deconv(deconv_white_box, 9, 3, 2, None, 'SAME')

        with tf.variable_scope("Classifier_PCA"):
            #pre_logit_pca = self.fs_layer(self.perturbated_data, 4096, activation=tf.nn.relu)
            pre_logit_pca = self.fs_layer(self.pca_input, 120, activation=tf.nn.relu)
            pre_logit_pca = self.fs_layer(pre_logit_pca, 84, activation=tf.nn.relu)
            self.logit_pca = self.fs_layer(pre_logit_pca, 6, activation=None)  
            self.prob_pca = tf.nn.softmax(self.logit_pca)

        with tf.variable_scope('Decoder_PCA'):
            #reshape_pca = tf.reshape(self.perturbated_data, shape=[-1, 4, 1, int(self.pca_dim/4)])
            reshape_pca = tf.reshape(self.pca_input, shape=[-1, 4, 1, int(self.pca_dim/4)])
            deconv_pca = self.deconv(reshape_pca, 64, 3, 4, tf.nn.relu, 'SAME')
            deconv_pca = self.deconv(deconv_pca, 32, 3, 2, tf.nn.relu, 'SAME')
            deconv_pca = self.deconv(deconv_pca, 16, 3, 2, tf.nn.relu, 'SAME')
            self.deconv_pca = self.deconv(deconv_pca, 9, 3, 2, None, 'SAME')


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

        ## while-box reconstruction attack, excluding the model inversion stage.
        self.theta_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Deep')
        self.theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Classifier')
        self.theta_r = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decov')

        self.loss_c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= self.label_p, logits=self.logit))
        self.loss_r = tf.losses.mean_squared_error(self.data_p, self.deconv_white_box)


        ### FOR PCA LEARNING NETWORKS.

        self.theta_c_pca = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Classifier_PCA')
        self.theta_d_pca = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decoder_PCA')

        self.loss_c_pca = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= self.label_p, logits=self.logit_pca))
        self.loss_r_pca = tf.losses.mean_squared_error(self.data_p, self.deconv_pca)

        uti_update = []
        scope = ["Deep", "Classifier"]
        for i in scope: 
            uti_update += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=i)
        with tf.control_dependencies(uti_update):

            op = tf.train.AdamOptimizer()
            self.opt = op.minimize(self.loss_c, var_list = self.theta_d + self.theta_c)

        uti_update_decov = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Decov")
        with tf.control_dependencies(uti_update_decov):  
            op_r = tf.train.AdamOptimizer()
            self.opt_r = op_r.minimize(self.loss_r, var_list = self.theta_r)

        uti_update_pca_d = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Decoder_PCA")
        with tf.control_dependencies(uti_update_pca_d):
            op_r_pca = tf.train.AdamOptimizer()
            self.opt_r_pca = op_r_pca.minimize(self.loss_r_pca, var_list = self.theta_d_pca)


        uti_update_pca_c = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Classifier_PCA")
        with tf.control_dependencies(uti_update_pca_c):
            op_c_pca = tf.train.AdamOptimizer()
            self.opt_c_pca = op_c_pca.minimize(self.loss_c_pca, var_list = self.theta_c_pca)


        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=5)

    def inject_noise(self, t_emb_pca, v_emb_pca):
        t_emb_pca = t_emb_pca + self.noise_factor * np.random.laplace(0, self.noise_scale, size=(t_emb_pca.shape[0], self.pca_dim))
        v_emb_pca = v_emb_pca + self.noise_factor * np.random.laplace(0, self.noise_scale, size=(v_emb_pca.shape[0], self.pca_dim))
        return t_emb_pca, v_emb_pca

    def train(self):

        ### Train the DNN first !! 
        epo = 1
        for _ in range(self.epo): 
            print("Epoch {} starts.".format(epo))
            for i, j in self.next_batch(self.t_data, self.t_label, self.batch_size, shuffle=True):
                sample = i.shape[0]
                feed_dict = {} 
                feed_dict[self.data_p] = i.reshape(sample, 128, 1, 9)
                feed_dict[self.label_p] = j
                _ = self.sess.run(self.opt, feed_dict=feed_dict)
            epo += 1
        ## Second, traing the deconv networks for intruding the deep features. 
        ## whilst computing the MSE.

        ## Get the embedding and its pca transformation. 
        t_emb = self.get_emb(self.t_data)
        v_emb = self.get_emb(self.v_data)

        pca = PCA(n_components=self.pca_dim)

        t_emb_pca = pca.fit_transform(t_emb)
        v_emb_pca = pca.transform(v_emb)

        t_emb_pca, v_emb_pca = self.inject_noise(t_emb_pca, v_emb_pca)

        train_matrix = self.get_train_matrix()
        print("************* Compute all weights *************")
        feed_dict[self.krr_weights_white_box_p] = self.KRR_close_form(t_emb, train_matrix, map=True)
        print("************* Compute all weights *************")

        feed_dict[self.lrr_weights_white_box_p] = self.LRR_close_form(t_emb, train_matrix)
        print("************* Compute all weights *************")

        feed_dict[self.krr_weights_pca_p] = self.KRR_close_form(t_emb_pca, train_matrix, map=True, pca=True)
        print("************* Compute all weights *************")

        feed_dict[self.lrr_weights_pca_p] = self.LRR_close_form(t_emb_pca, train_matrix, pca=True)
        print("************* Compute all weights *************")


        self.sess.run(self.assign_op, feed_dict=feed_dict)

        print("********** Evaluation of the deep features **********")

        for _ in range(self.epo):
            for i, j, k in self.eva_next_batch(t_emb, self.t_data, self.t_label, self.batch_size, shuffle=False):
                sample = i.shape[0]
                no  = np.random.normal(size=(sample, self.com_dim))
                feed_dict = {} 
                feed_dict[self.com_input_p] = i
                feed_dict[self.data_p] = j.reshape(sample, 128, 1, 9)
                _ = self.sess.run(self.opt_r, feed_dict=feed_dict)

        mse, mse_lrr, mse_krr = self.prediction_and_mse(t_emb, v_emb, self.v_label, PCA=False)
        acc = self.prediction_and_accuracy(v_emb, self.v_label, PCA=False)

        print("********** Evaluation of the PCA features **********")

        for _ in range(self.epo):
            for i, j, k  in self.eva_next_batch(t_emb_pca, self.t_data, self.t_label, self.batch_size, shuffle=False):
                sample = i.shape[0]
                no  = np.random.laplace(size=(sample, self.pca_dim))
                feed_dict = {} 
                feed_dict[self.pca_input] = i
                feed_dict[self.data_p] = j.reshape(sample, 128, 1, 9)
                feed_dict[self.label_p] = k
                feed_dict[self.pca_noise] =  no 
                _, _ = self.sess.run([self.opt_r_pca, self.opt_c_pca], feed_dict=feed_dict)

        mse_pca, mse_pca_lrr, mse_pca_krr = self.prediction_and_mse(t_emb_pca, v_emb_pca, self.v_label, PCA=True)
        acc_pca = self.prediction_and_accuracy(v_emb_pca, self.v_label, PCA=True)

        return acc, mse, mse_lrr, mse_krr, acc_pca, mse_pca, mse_pca_lrr, mse_pca_krr

    def get_emb(self, data):
        
        temp = []
        for i, j in self.next_batch(data, self.t_label, self.batch_size, shuffle=False):
                feed_dict = {} 
                feed_dict[self.data_p] = i.reshape(-1, 128, 1, 9)
                #feed_dict[self.label_p] = j.reshape(-1)      
                temp.append(self.sess.run(self.compressing, feed_dict=feed_dict))
        return np.concatenate(temp, axis=0)

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

    def next_batch(self, t_data, t_label, batch_size, shuffle=False, MNIST_emb = False):

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
                if MNIST_emb : 
                    yield np.array(t_data[i:]) , np.array(t_label[i:])
                else: 
                    yield np.array(t_data[i:]) , np.array(t_label[i:])
            else : 
                if MNIST_emb: 
                    yield np.array(t_data[i: i+self.batch_size]) , np.array(t_label[i: i+self.batch_size])
                else:
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

        for i, j in self.next_batch(data, label, self.batch_size, MNIST_emb = True):

            sample = i.shape[0]
            feed_dict = {} 

            if PCA: 
                no  = np.random.laplace(size=(sample, self.pca_dim))
                feed_dict[self.pca_input] = i
                feed_dict[self.pca_noise] = no 
                prob = self.sess.run(self.prob_pca, feed_dict=feed_dict)

            else : 

                feed_dict[self.compressing] = i
                prob = self.sess.run(self.prob, feed_dict=feed_dict)

            pred.append(prob)
        predict = np.concatenate(pred, axis=0)
        predict = np.argmax(predict, axis=1)
        y_true = np.argmax(self.v_label, axis=1)
        accuracy = accuracy_score(predict, y_true)

        if PCA :
            print("PCA with noise Accuracy: {}".format(accuracy))      
        else: 
            print("Deep features Accuracy: {}".format(accuracy))

        return accuracy

    def prediction_and_mse(self, train_data, data, label, PCA=True): 

        pred = []
        predict_lrr = []
        predict_krr = []
        for i, j in self.next_batch(data, label, self.batch_size, MNIST_emb=True):

            sample = i.shape[0]
            feed_dict = {} 

            if PCA : 
                no  = np.random.laplace(size=(sample, self.pca_dim))
                feed_dict[self.pca_input] = i
                feed_dict[self.pca_noise] = no
                reco, reco_lrr, reco_krr  = self.sess.run([self.deconv_pca, self.upsampling_pca_lrr, self.upsampling_pca_krr], feed_dict=feed_dict)
            else :
                feed_dict[self.com_input_p] = i
                reco, reco_lrr, reco_krr = self.sess.run([self.deconv_white_box, self.upsampling_white_box_lrr, self.upsampling_white_box_krr], feed_dict=feed_dict)
            pred.append(reco)
            predict_lrr.append(reco_lrr)
            predict_krr.append(reco_krr)
        predict = np.concatenate(pred, axis=0)

        predict = np.concatenate(pred, axis=0)
        predict_lrr = np.concatenate(predict_lrr, axis=0)
        predict_krr = np.concatenate(predict_krr, axis=0)

        error = [] 
        error_lrr = []
        error_krr = []

        for i in range(len(predict)):
            error.append(mean_squared_error(self.v_data[i].flatten(), predict[i].flatten())*9)
            error_lrr.append(mean_squared_error(self.v_data[i].flatten(), predict_lrr[i].flatten())*9)
            error_krr.append(mean_squared_error(self.v_data[i].flatten(), predict_krr[i].flatten())*9)

        #print(error_lrr[0])
        #print(error_krr[0])
        mse = np.mean(error)
        mse_lrr = np.mean(error_lrr)
        mse_krr= np.mean(error_krr)

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
                for i, j in self.next_batch(emb, self.t_label, self.batch_size, MNIST_emb=True):
                    feed_dict = {}
                    sample = i.shape[0]
                    feed_dict[self.pca_input] = i
                    feed_dict[self.pca_noise] =  np.random.laplace(size=(sample, self.pca_dim))
                    temp.append(self.sess.run(self.pca_map, feed_dict=feed_dict))
                emb_matrix = np.concatenate(temp, axis=0)

            else : 

                a,b = emb.shape
                no = np.random.laplace(size=(a, b))
                #emb_matrix = emb + self.noise_factor * no
                emb_matrix = emb 

                #emb_matrix = emb_matrix + (self.noise_factor * no)
        else : 

            if map: 
                for i, j in self.next_batch(emb, self.t_label, self.batch_size, MNIST_emb=True):
                    feed_dict = {}
                    sample = i.shape[0]
                    feed_dict[self.com_input_p] = i
                    temp.append(self.sess.run(self.com_map, feed_dict=feed_dict))
                emb_matrix = np.concatenate(temp, axis=0)

            else : 
                emb_matrix = emb
        rau = 0.001
        
        mu = np.mean(emb_matrix, axis=0)
        emb_matrix = emb_matrix #- mu
    
        emb_matrix = emb_matrix.T 
        s = np.dot(emb_matrix, emb_matrix.T)
        a,b = s.shape
        identity = np.identity(a)
        s_inv = np.linalg.inv(s + rau * np.identity(a))
        train_mu = np.mean(train_matrix, axis=0)
        train_norm = train_matrix #- train_mu
        #weights = np.dot(np.dot(s_inv,emb_matrix), train_norm)
        weights = np.dot(np.dot(s_inv, emb_matrix), train_norm)


        #clf = Ridge(alpha=0.001, fit_intercept=False, random_state=9)
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
                for i, j in self.next_batch(emb, self.t_label, self.batch_size, MNIST_emb=True):
                    feed_dict = {}
                    sample = i.shape[0]
                    feed_dict[self.pca_input] = i
                    feed_dict[self.pca_noise] = np.random.laplace(size=(sample, self.pca_dim))
                    temp.append(self.sess.run(self.pca_map, feed_dict=feed_dict))
                emb_matrix = np.concatenate(temp, axis=0)
            else : 
                a, b = emb.shape
                no = np.random.laplace(size=(a, b))
                #emb_matrix = emb + self.noise_factor* no
                emb_matrix = emb
                #emb_matrix = emb_matrix + (self.noise_factor * no)
        else : 
            if map : 
                for i, j in self.next_batch(emb, self.t_label, self.batch_size, MNIST_emb=True):
                    feed_dict = {}
                    sample = i.shape[0]
                    feed_dict[self.com_input_p] = i
                    temp.append(self.sess.run(self.com_map, feed_dict=feed_dict))
                emb_matrix = np.concatenate(temp, axis=0)
            else : 
                emb_matrix = emb
        mu = np.mean(emb_matrix, axis=0)
        #print("Shape of mu: {}.".format(mu.shape))
        emb_matrix = emb_matrix #- mu
        emb_matrix = emb_matrix.T
        rau = 0.001
        s = np.dot(emb_matrix, emb_matrix.T)
        h,w = s.shape
        s_inv = np.linalg.inv(s + rau * np.identity(h))    
        train_mu = np.mean(train_matrix, axis=0)
        train_norm = train_matrix #- train_mu
        #clf = Ridge(alpha=0.001, fit_intercept=False, random_state=9)
        #clf.fit(emb_matrix, train_norm)
        #clf.fit(emb_matrix, train_matrix)
        #weights = clf.coef_ 
        weights = np.dot(np.dot(s_inv, emb_matrix), train_norm)
        #weights = np.dot(np.dot(s_inv, emb_matrix), train_matrix)
        print('Shape of LRR weights: {}'.format(weights.shape))
        return weights





        

