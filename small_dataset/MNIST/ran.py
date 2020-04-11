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
np.random.seed(9)
random.seed(9)
#tf.set_random_seed(9)
import math
class RAN:
    def __init__(self, arg):

        #np.random.seed(9)
        #random.seed(9)
        #tf.set_random_seed(9)
        self.arg = arg 
        self.trade_off = self.arg.trade_off
        self.com_dim = 400 #256
        self.epo = self.arg.epoch
        self.mapping_dim = self.arg.mapping_dim
        self.gamma = self.arg.gamma
        self.seed = self.arg.seed
        self.batch_size = self.arg.batch_size
        self.t_data, self.t_label, self.v_data, self.v_label= self.load_data()
        self.RAN()

    def load_data(self):

        mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
        
        t_data  = (mnist.train.images*255)
        t_data = (t_data/127.5) - 1
        t_label = mnist.train.labels
        v_data = mnist.test.images * 255 
        v_data = (v_data/127.5) -1
        v_label = mnist.test.labels
        '''
        t_data  = mnist.train.images 
        t_label = mnist.train.labels
        v_data = mnist.test.images
        v_label = mnist.test.labels
        '''
        return t_data, t_label, v_data, v_label

    def deconv(self, x, kernel_size, stride, activetion, pad): 
        return ly.conv2d_transpose(x, kernel_size=k, stride=(stride, 1), activation_fn=activetion, padding=pad, weights_initializer=tf.contrib.layers.xavier_initializer())

    def conv(self, x, kernel_size, stride, activetion, pad): 
        return ly.conv2d(x, kernel_size=k, stride=(stride, 1), activation_fn=activetion, padding=pad, weights_initializer=tf.contrib.layers.xavier_initializer())

    def fs_layer(self, x, units, activation): 
        return ly.fully_connected(x, units, activation_fn=activation, weights_initializer=tf.contrib.layers.xavier_initializer())

    def Alex_net(self, input, name_1, name_2):

        with tf.variable_scope(name_1):

            conv1 = ly.conv2d(input, 96, kernel_size=11, stride=4, padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            #conv1 = self.bo_batch_norm(conv1, self.is_train)
            conv1 = ly.max_pool2d(conv1, kernel_size=3, stride=2, padding='SAME')

            conv2 = ly.conv2d(conv1, 256, kernel_size=5, stride=1, padding='SAME', activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer())
            #conv2 = self.bo_batch_norm(conv2, self.is_train)
            conv2 = ly.max_pool2d(conv2, kernel_size=3, stride=2, padding='SAME')

            conv3 = ly.conv2d(conv2, 384, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer())
            #conv3 = self.bo_batch_norm(conv3, self.is_train)
            #conv3 = ly.max_pool2d(conv3,kernel_size=3,stride=2,padding='SAME')

            conv4 = ly.conv2d(conv3, 384, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer())
            #conv4 = self.bo_batch_norm(conv4, self.is_train)
            #conv4 = ly.max_pool2d(conv4,kernel_size=3,stride=2,padding='SAME')

            conv5 = ly.conv2d(conv4, 256, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer())
            #conv5 = self.bo_batch_norm(conv5, self.is_train)
            conv5 = ly.max_pool2d(conv5, kernel_size=3, stride=2, padding='SAME')
            flat = ly.flatten(conv5)
            print(conv5)

        with tf.variable_scope(name_2):

            fc1 = ly.fully_connected(flat, 4096, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            fc2 = ly.fully_connected(fc1, 4096, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            logit = ly.fully_connected(fc2, 10, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())

        return flat, logit

    def LeNet(self, input, name_1, name_2):

        with tf.variable_scope(name_1):
            conv1 = ly.conv2d(input, 6, kernel_size=5, stride=1, padding='SAME', activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer())
            conv1 = ly.max_pool2d(conv1, kernel_size=2, stride=2, padding='VALID')

            conv2 = ly.conv2d(conv1, 16, kernel_size=5, stride=1, padding='VALID', activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer())
            conv2 = ly.max_pool2d(conv2, kernel_size=2, stride=2, padding='VALID')

            #conv3 = ly.conv2d(conv2, 120, kernel_size=5, stride=1, padding='VALID', activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer())
            #conv3 = ly.max_pool2d(conv3, kernel_size=2, stride=2, padding='VALID')
            flat = ly.flatten(conv2) ### dimension 400 
            print(flat) ### dimension 400
        with tf.variable_scope(name_2):

            fc1 = ly.fully_connected(flat, 120, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            fc2 = ly.fully_connected(fc1, 84, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            logit = ly.fully_connected(fc2, 10, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())

        return flat, logit

    def adversary_krr(self, compressive_data, reuse=False):
        #final_latent = tf.concat([latent, latent_1], axis=1)
        with tf.variable_scope('adversary_krr') as scope:  
            if reuse: 
                scope.reuse_variables()
            recontruction = ly.fully_connected(compressive_data, 28*28*1, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = None)
        #return tf.reshape(recontruction, shape=[-1, 175, 175, 3])
        return tf.reshape(recontruction, shape=[-1, 28, 28, 1])

    def adversary_lrr(self, compressive_data, reuse=False):
        #final_latent = tf.concat([latent, latent_1], axis=1)
        with tf.variable_scope('adversary_lrr') as scope:  
            if reuse: 
                scope.reuse_variables()
            recontruction = ly.fully_connected(compressive_data, 28*28*1, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = None)
        #return tf.reshape(recontruction, shape=[-1, 175, 175, 3])
        return tf.reshape(recontruction, shape=[-1, 28, 28, 1])

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

        '''
        omega_matrix = constant_op.constant(np.sqrt(2 * gamma) *
           random_state.normal(size=omega_matrix_shape),dtype=dtypes.float32)

        bias = constant_op.constant(
            random_state.uniform(
            0.0, 2 * np.pi, size=bias_shape), dtype=dtypes.float32)

        x_omega_plus_bias = math_ops.add(
            math_ops.matmul(input_tensor, omega_matrix), bias)
        '''

        return math.sqrt(2.0 / output_dim) * math_ops.cos(x_omega_plus_bias)

    def RAN(self):

        self.data_p = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        self.label_p = tf.placeholder(tf.int64, shape=[None])
        self.one_hot = tf.one_hot(self.label_p, 10)
        self.emb_p = tf.placeholder(tf.float32, shape=[None, self.com_dim])
        #noise_p = tf.placeholder(tf.float32, shape=[None, self.com_dim])
        self.keep_prob = tf.placeholder(tf.float32)


        #self.compressing, self.logit = self.Alex_net(self.data_p, "Encoder", "Classifier")
        self.compressing, self.logit = self.LeNet(self.data_p, "Encoder", "Classifier")
        self.prob = tf.nn.softmax(self.logit)

        self.krr_weights_p = tf.placeholder(tf.float32, shape=[self.mapping_dim, 28*28*1])
        self.lrr_weights_p = tf.placeholder(tf.float32, shape=[self.com_dim, 28*28*1])

        self.emb_map = self.RFF_map(self.emb_p, self.seed, self.gamma, self.com_dim, self.mapping_dim)
        self.krr_reco = self.adversary_krr(self.emb_map)
        self.lrr_reco = self.adversary_lrr(self.emb_p)

        self.theta_krr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='adversary_krr')
        self.theta_lrr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='adversary_lrr')

        self.assign_op = []
        self.assign_op.append(self.theta_krr[0].assign(self.krr_weights_p))
        self.assign_op.append(self.theta_lrr[0].assign(self.lrr_weights_p))

        with tf.variable_scope('white_box'):
            latent = ly.fully_connected(self.emb_p, 7*7*64, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            pre_reshape = tf.reshape(latent, shape=[-1, 7, 7, 64])
            #upsampling_white_box = ly.conv2d_transpose(pre_reshape, 64, kernel_size=3, stride=2, padding='SAME', weights_initializer=tf.contrib.layers.xavier_initializer())
            upsampling_white_box = ly.conv2d_transpose(pre_reshape, 32, kernel_size=3, stride=2, padding='SAME', weights_initializer=tf.contrib.layers.xavier_initializer())
            self.recon_white_box = ly.conv2d_transpose(upsampling_white_box, 1, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer())


        uti_update_white_box = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='white_box')
        with tf.control_dependencies(uti_update_white_box):
            self.theta_white_box = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='white_box')
            self.loss_white_box = tf.losses.mean_squared_error(self.data_p, self.recon_white_box)
            self.opt_white_box = tf.train.AdamOptimizer().minimize(self.loss_white_box, var_list= self.theta_white_box)

        with tf.variable_scope("Decoder"):
            latent = ly.fully_connected(self.compressing, 7*7*64, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            pre_reshape = tf.reshape(latent, shape=[-1, 7, 7, 64])
            self.upsampling = ly.conv2d_transpose(pre_reshape, 32, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            self.upsampling = ly.conv2d_transpose(self.upsampling, 1, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer())

        self.loss_pri = tf.losses.mean_squared_error(self.data_p, self.upsampling) 
        self.loss_uti = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= self.one_hot, logits= self.logit))

        self.loss_g = self.trade_off * self.loss_uti - (1 - self.trade_off) * self.loss_pri
        
        theta_e = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encoder')

        theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Classifier')

        theta_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decoder')

        uti_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


        self.gradients_d = tf.gradients(self.loss_pri, theta_e)
        gradient_norm = []
        print(self.gradients_d)
        for i in self.gradients_d: 
            print(i)
            print(len(i.get_shape().as_list()))
            if len(i.get_shape().as_list()) == 1:
                #gradient_norm.append(tf.norm(i, ord=2))
                grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(i), axis=[0]))
                gradient_norm.append(grad_l2)

            else : 
                #gradient_norm.append(tf.norm(i, ord=2))
                grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(i), axis=[0,1,2,3]))
                gradient_norm.append(grad_l2)

        self.gradient_norm_d = tf.reduce_sum(gradient_norm)

        self.gradients_c = tf.gradients(self.loss_uti, theta_e)
        gradient_norm = []
        print(self.gradients_c)
        for i in self.gradients_c: 
            print(i)
            #for j in i : 
            print(len(i.get_shape().as_list()))
            if len(i.get_shape().as_list()) == 1:
                #gradient_norm.append(tf.norm(i, ord=2))
                grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(i), axis=[0]))
                gradient_norm.append(grad_l2)

            else : 
                #gradient_norm.append(tf.norm(i, ord=2))
                grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(i), axis=[0,1,2,3]))
                gradient_norm.append(grad_l2)


        self.gradients_cls = tf.gradients(self.loss_uti, theta_c)
        #gradient_norm = []
        print(self.gradients_cls)
        for i in self.gradients_cls: 
            print(i)
            print(len(i.get_shape().as_list()))
            if len(i.get_shape().as_list()) == 1:
                #gradient_norm.append(tf.norm(i, ord=2))
                grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(i), axis=[0]))
                gradient_norm.append(grad_l2)

            elif len(i.get_shape().as_list()) == 2:
                #gradient_norm.append(tf.norm(i, ord=2))
                grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(i), axis=[0, 1]))
                gradient_norm.append(grad_l2)

            else : 
                #gradient_norm.append(tf.norm(i, ord=2))
                grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(i), axis=[0,1,2,3]))
                gradient_norm.append(grad_l2)


        self.gradient_norm_c = tf.reduce_sum(gradient_norm)

        scope = ["Encoder", "Classifier", "Decoder"]
        uti_update = []
        for i in scope : 
            uti_update += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=i)
        with tf.control_dependencies(uti_update):
            e_c_op = tf.train.AdamOptimizer(0.001)
            self.e_c_opt = e_c_op.minimize(self.loss_uti, var_list=theta_e + theta_c)

            d_op = tf.train.AdamOptimizer(0.001)
            self.d_opt = d_op.minimize(self.loss_pri, var_list=theta_d)

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
                yield np.array(t_data[i:]).reshape(-1, 28, 28, 1), np.array(t_label[i:])
            else : 
                yield np.array(t_data[i: i+self.batch_size]).reshape(-1, 28, 28, 1), np.array(t_label[i: i+self.batch_size])

    def train(self):
        epo = 1
        grad_record_c = []
        grad_record_d = []
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
                _, grad_c, grad_d = self.sess.run([self.g_opt, self.gradient_norm_c, self.gradient_norm_d], feed_dict=feed_dict)
                grad_record_d.append(grad_d)
                grad_record_c.append(grad_c)
                print(grad_c, grad_d)
            epo +=1

        np.save("grad_c.npy", grad_record_c)
        np.save("grad_d.npy", grad_record_d)
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
            prob = self.sess.run(self.prob, feed_dict=feed_dict)
            pred.append(prob)

        predict = np.concatenate(pred, axis=0)
        predict = np.argmax(predict, axis=1)
        accuracy = accuracy_score(predict, np.array(self.v_label))       
        print("Deep features Accuracy: {}".format(accuracy))

        return accuracy
    '''
    def plot(self, x):
        x = x - np.min(x)
        x /= np.max(x)
        x *= 255  
        x= x.astype(np.uint8)
        x = x.reshape(28,28,1)
        return x 
    '''
    

    def plot(self, x):
        #cifar_mean = np.array([0.4914, 0.4822, 0.4465])
        #cifar_std = np.array([0.2470, 0.2435, 0.2616])
        x = self.inverse_transform(x)
        #x = x - np.min(x)
        #x /= np.max(x)
        x *= 255  
        x= x.astype(np.uint8)
        x = x.reshape(28, 28, 1)
        return x 

    def inverse_transform(self, x):
        x = x+1 
        x = x/2 
        return x 

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
                #print(emb.shape)
                rff_map = self.sess.run(self.emb_map, feed_dict = {self.emb_p:emb}) 
                #print(rff_map.shape)

                #feed_dict[self.emb_p] = emb
                #feed_dict[self.noise_p] = self.factor * no
                _ = self.sess.run(self.opt_white_box, feed_dict={self.emb_p:emb, self.data_p:i})

                if epo == 1 : 
                    data_list.append(i)
                    emb_list.append(emb)
                    map_list.append(rff_map)
            epo+=1

        emb = np.concatenate(emb_list, axis=0)
        #print(emb.shape)
        rff_map = np.concatenate(map_list, axis=0)
        #print(rff_map.shape)
        feed_dict = {}
        feed_dict[self.krr_weights_p] = self.KRR_close_form(rff_map, self.t_data)
        feed_dict[self.lrr_weights_p] = self.LRR_close_form(emb, self.t_data)
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
            reco, reco_lrr, reco_krr = self.sess.run([self.recon_white_box, self.lrr_reco, self.krr_reco], feed_dict=feed_dict)
            for k in range(len(reco)):
                mse_list.append(mean_squared_error(self.plot(i[k]).flatten(), self.plot(reco[k]).flatten()))
                mse_list_lrr.append(mean_squared_error(self.plot(i[k]).flatten(), self.plot(reco_lrr[k]).flatten()))
                mse_list_krr.append(mean_squared_error(self.plot(i[k]).flatten(), self.plot(reco_krr[k]).flatten()))

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









    