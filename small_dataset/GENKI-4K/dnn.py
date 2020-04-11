from sklearn.decomposition import PCA ## For resize. 
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
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from sklearn.utils import check_random_state
from sklearn.kernel_approximation import RBFSampler
import math

from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from scipy.misc import imread, imsave, imresize

class DNN:

    def __init__(self, arg):
        #np.random.seed(9)
        #random.seed(9)
        #tf.set_random_seed(9)

        self.arg = arg 
        self.path = self.arg.path
        self.com_dim = 3136 #400 #1024 #3136
        self.noise_factor = self.arg.noise_term
        self.noise_scale = self.arg.noise_scale
        self.pca_dim = self.arg.pca_dim
        self.epo = self.arg.epoch
        #self.pca_dim = self.arg.pca_dim
        self.batch_size = self.arg.batch_size
        self.seed = int(self.arg.seed)
        self.gamma = self.arg.gamma
        self.mapping_dim = self.arg.mapping_dim
        self.mapping_dim_pca = self.arg.mapping_dim_pca
        self.t_data, self.t_label, self.v_data, self.v_label= self.load_data()
        self.DNN_with_resize()


    def load_data(self):
        folder = os.listdir(self.path)

        train_folder = os.path.join(self.path, folder[2])
        test_folder = os.path.join(self.path, folder[0])

        label_list = ['0', '1']

        t_data = []
        t_label = []
        for i in label_list:
            path = os.path.join(train_folder, i)
            img_name = sorted(os.listdir(path))
            for k in range(len(img_name)):
                t_data.append(imread(os.path.join(path, img_name[k])).reshape(1, 64, 64, 3))
                t_label.append(int(i))

        v_data = []
        v_label = []
        for i in label_list:
            path = os.path.join(test_folder, i)
            img_name = sorted(os.listdir(path))
            for k in range(len(img_name)):
                v_data.append(imread(os.path.join(path, img_name[k])).reshape(1, 64, 64, 3))
                v_label.append(int(i))      

        t_data = (np.concatenate(t_data, axis=0)/127.5) - 1
        t_label = np.array(t_label)

        v_data = (np.concatenate(v_data, axis=0)/127.5) - 1
        v_label = np.array(v_label)

        #all_data = t_data + v_data 
        #all_label = t_label + v_label
        ### Shuffle then sample ...
        #c = list(zip(all_data, all_label))
        #random.shuffle(c)
        #all_data, all_label = zip(*c)

        #t_data, t_label, v_data, v_label = self.sample(all_data, all_label)

        #print(t_data.shape)
        #print(t_label.shape)
        #print(v_data.shape)
        #print(v_label.shape)

        return t_data, t_label, v_data, v_label

    def sample(self, all_data, all_label):

        positive_list = []
        negative_list = []

        for i in range(len(all_label)):
            if all_label[i] == 1:
                positive_list.append(i)
            else: 
                negative_list.append(i)


        t_positive_indice, v_positive_indice = train_test_split(positive_list, test_size=0.1, random_state=9)
        t_negative_indice, v_negative_indice = train_test_split(negative_list, test_size=0.1, random_state=9)

        all_data = np.concatenate(all_data, axis=0)
        all_label = np.array(all_label)


        t_data = np.concatenate([all_data[np.array(t_positive_indice), :], all_data[np.array(t_negative_indice), :]], axis=0)
        v_data = np.concatenate([all_data[np.array(v_positive_indice), :], all_data[np.array(v_negative_indice), :]], axis=0)


        t_label = np.concatenate([all_label[[np.array(t_positive_indice)]], all_label[[np.array(t_negative_indice)]]], axis=0)
        v_label = np.concatenate([all_label[[np.array(v_positive_indice)]], all_label[[np.array(v_negative_indice)]]], axis=0)

        return t_data, t_label, v_data, v_label

    def fs_layer(self, x, units, activation): 
        return ly.fully_connected(x, units, activation_fn=activation, weights_initializer=tf.contrib.layers.xavier_initializer())



    def Alex_net(self, input, name_1, name_2):

        with tf.variable_scope(name_1):

            conv1 = ly.conv2d(input, 96, kernel_size=11, stride=4, padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            #conv1 = self.bo_batch_norm(conv1, self.is_train)
            conv1 = ly.max_pool2d(conv1, kernel_size=3, stride=2, padding='VALID')

            conv2 = ly.conv2d(conv1, 256, kernel_size=5, stride=1, padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            #conv2 = self.bo_batch_norm(conv2, self.is_train)
            conv2 = ly.max_pool2d(conv2, kernel_size=3, stride=2, padding='VALID')

            conv3 = ly.conv2d(conv2, 384, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            #conv3 = self.bo_batch_norm(conv3, self.is_train)
            #conv3 = ly.max_pool2d(conv3,kernel_size=3,stride=2,padding='SAME')

            conv4 = ly.conv2d(conv3, 384, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            #conv4 = self.bo_batch_norm(conv4, self.is_train)
            #conv4 = ly.max_pool2d(conv4,kernel_size=3,stride=2,padding='SAME')

            conv5 = ly.conv2d(conv4, 256, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            #conv5 = self.bo_batch_norm(conv5, self.is_train)
            conv5 = ly.max_pool2d(conv5, kernel_size=3, stride=2, padding='VALID')
            flat = ly.flatten(conv5)
            print(conv5)
            #flat = ly.fully_connected(flat, 512, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())

        with tf.variable_scope(name_2):

            fc1 = ly.fully_connected(flat, 4096, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            fc2 = ly.fully_connected(fc1, 4096, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            logit = ly.fully_connected(fc2, 2, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())

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
            logit = ly.fully_connected(fc2, 2, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())

        return flat, logit


    def vgg13(slef,input, name_1, name_2):

        with tf.variable_scope(name_1):
            conv1 = ly.conv2d(input, 16, kernel_size=3, stride=1, padding='SAME', activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer())
            conv1 = ly.max_pool2d(conv1, kernel_size=3, stride=2, padding='SAME')

            conv2 = ly.conv2d(conv1, 16, kernel_size=3, stride=1, padding='SAME', activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer())
            conv2 = ly.max_pool2d(conv2, kernel_size=3, stride=2, padding='SAME')

            conv3 = ly.conv2d(conv2, 16, kernel_size=3, stride=1, padding='SAME', activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer())
            conv3 = ly.max_pool2d(conv3, kernel_size=3, stride=2, padding='SAME')
            flat = ly.flatten(conv3) ### dimension 1024
            flat = ly.fully_connected(flat, 400, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            print(flat) ### dimension 400

        with tf.variable_scope(name_2):
            fc1 = ly.fully_connected(flat, 128, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            fc2 = ly.fully_connected(fc1, 128, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            logit = ly.fully_connected(fc2, 2, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())

        return flat, logit

    def init_tensor(self, shape):

        return tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=1.0))

    def adversary_krr(self, kernel_vec, name, reuse=False):
        #final_latent = tf.concat([latent, latent_1], axis=1)
        with tf.variable_scope('adversary_krr_'+name) as scope:  
            if reuse: 
                scope.reuse_variables()
            recontruction = ly.fully_connected(kernel_vec, 64*64*3, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = None)
        #return tf.reshape(recontruction, shape=[-1, 175, 175, 3])
        return tf.reshape(recontruction, shape=[-1, 64, 64, 3])


    def adversary_lrr(self, compressive_data, name, reuse=False):
        #final_latent = tf.concat([latent, latent_1], axis=1)
        with tf.variable_scope('adversary_lrr_'+name) as scope:  
            if reuse: 
                scope.reuse_variables()
            recontruction = ly.fully_connected(compressive_data, 64*64*3, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = None)
        #return tf.reshape(recontruction, shape=[-1, 175, 175, 3])
        return tf.reshape(recontruction, shape=[-1, 64, 64, 3])

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

    def bo_batch_norm(self,x, is_training, momentum=0.9, epsilon=0.0001):

        x = tf.layers.batch_normalization(x, momentum=momentum, epsilon=epsilon,training=is_training)
        x = tf.nn.relu(x)

        return x

    def DNN_with_resize(self):

        self.data_p = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
        self.label_p = tf.placeholder(tf.int64, shape=[None])
        self.one_hot = tf.one_hot(self.label_p, 2)
        self.is_train = tf.placeholder(tf.bool)
        #self.noise_p = tf.placeholder(tf.float32, shape=[None, self.ori_dim])

        self.com_input_p = tf.placeholder(tf.float32, shape=[None, self.com_dim])
        #self.kernel_vec_p = tf.placeholder(tf.float32, shape=[None, len(self.t_data)])
        self.pca_input = tf.placeholder(tf.float32, shape=[None, self.pca_dim])
        self.pca_noise = tf.placeholder(tf.float32, shape=[None ,self.pca_dim])

        self.perturbated_data = tf.add(self.pca_input, self.noise_factor * self.pca_noise)

        self.com_map = self.RFF_map(self.com_input_p, self.seed, self.gamma, self.com_dim, self.mapping_dim)
        self.pca_map = self.RFF_map(self.perturbated_data, self.seed, self.gamma, self.pca_dim, self.mapping_dim_pca)

        #self.compressing, self.logit = self.Alex_net(self.data_p, "Deep", "Classifier")
        self.compressing, self.logit = self.LeNet(self.data_p, "Deep", "Classifier")
        #self.compressing, self.logit = self.vgg13(self.data_p, "Deep", "Classifier")


        self.prob = tf.nn.softmax(self.logit)

        self.krr_weights_white_box_p = tf.placeholder(tf.float32, shape=[self.mapping_dim, 64*64*3])
        self.lrr_weights_white_box_p = tf.placeholder(tf.float32, shape=[self.com_dim, 64*64*3])
        
        self.krr_weights_pca_p = tf.placeholder(tf.float32, shape=[self.mapping_dim_pca, 64*64*3])
        self.lrr_weights_pca_p = tf.placeholder(tf.float32, shape=[self.pca_dim, 64*64*3])

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


        with tf.variable_scope("Decov"):
            #reshape_decov = tf.reshape(self.com_input_p, shape=[-1, 4, 4, 64])
            latent = ly.fully_connected(self.com_input_p, 4*4*256, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            latent = tf.reshape(latent, shape=[-1, 4, 4, 256])
            upsampling = ly.conv2d_transpose(latent, 128, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            #upsampling = self.bo_batch_norm(upsampling, self.is_train)
            upsampling = ly.conv2d_transpose(upsampling, 64, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())   
            #upsampling = self.bo_batch_norm(upsampling, self.is_train)  
            upsampling = ly.conv2d_transpose(upsampling, 32, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            #upsampling = self.bo_batch_norm(upsampling, self.is_train)
            self.upsampling = ly.conv2d_transpose(upsampling, 3, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer())
       

        ## PCA Stage.

        with tf.variable_scope("Classifier_PCA"):
            fs_pca = ly.fully_connected(self.pca_input, 128, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            fs_pca = ly.fully_connected(fs_pca, 128, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            self.logit_pca = ly.fully_connected(fs_pca, 2, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
            self.prob_pca = tf.nn.softmax(self.logit_pca)

        with tf.variable_scope("Decoder_PCA"):
            #latent_pca = tf.reshape(self.perturbated_data, shape=[-1, 4, 4, 32])
            latent_pca = ly.fully_connected(self.pca_input, 4*4*256, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            latent_pca = tf.reshape(latent_pca, shape=[-1, 4, 4, 256])
            upsampling_pca = ly.conv2d_transpose(latent_pca, 128, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            upsampling_pca = ly.conv2d_transpose(upsampling_pca, 64, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            upsampling_pca = ly.conv2d_transpose(upsampling_pca, 32, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            self.upsampling_pca = ly.conv2d_transpose(upsampling_pca, 3, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer())
       

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
            op = tf.train.AdamOptimizer()
            self.opt = op.minimize(self.loss_c, var_list = self.theta_d + self.theta_c)

        uti_update_decov = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Decov")
        with tf.control_dependencies(uti_update_decov):        
            op_r = tf.train.AdamOptimizer()
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
        self.saver = tf.train.Saver(max_to_keep=5)

        ## First end-to-end train the deep model while computing the accuracy.

    def inject_noise(self, t_emb_pca, v_emb_pca):
        t_emb_pca = t_emb_pca + self.noise_factor * np.random.laplace(0, self.noise_scale, size=(t_emb_pca.shape[0], self.pca_dim))
        v_emb_pca = v_emb_pca + self.noise_factor * np.random.laplace(0, self.noise_scale, size=(v_emb_pca.shape[0], self.pca_dim))
        return t_emb_pca, v_emb_pca

    def train(self):

        count = 1

        for _ in range(self.epo): 
            print("Epoch {} starts.".format(count))
            for i, j in self.next_batch(self.t_data, self.t_label, self.batch_size, shuffle=True):
                sample = i.shape[0]
                #no  = np.random.normal(size=(sample, self.com_dim))
                feed_dict = {} 
                feed_dict[self.data_p] = i
                feed_dict[self.label_p] = j
                _ = self.sess.run(self.opt, feed_dict=feed_dict)
            count += 1
        ## Second, traing the devoc networks for the deep feature 
        ## while computing the MSE.
        ## tune the parameters....
        t_emb = self.get_emb(self.t_data)
        v_emb = self.get_emb(self.v_data)

        pca = PCA(n_components = self.pca_dim, random_state=9)

        t_emb_pca = pca.fit_transform(t_emb)
        v_emb_pca = pca.transform(v_emb)

        t_emb_pca, v_emb_pca = self.inject_noise(t_emb_pca, v_emb_pca)

        train_matrix = self.get_train_matrix()
        ### may be incorporated in one function.

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

        for _ in range(100):
            for i, j, k in self.eva_next_batch(t_emb, self.t_data, self.t_label, self.batch_size, shuffle=False):
                sample = i.shape[0]
                #no = np.random.multivariate_normal([0 for i in range(self.com_dim)], np.identity(self.com_dim), sample)
                #no  = np.random.normal(size=(sample, self.com_dim))
                feed_dict = {} 
                feed_dict[self.com_input_p] = i
                feed_dict[self.data_p] = j 
                feed_dict[self.is_train] = True
                _ = self.sess.run(self.opt_r, feed_dict=feed_dict)

        mse, mse_lrr, mse_krr = self.prediction_and_mse(t_emb, v_emb, self.v_label, PCA=False)
        acc = self.prediction_and_accuracy(v_emb, self.v_label, PCA=False)

        print("********** Evaluation of the PCA features **********")

        for _ in range(100):
            for i, j, k in self.eva_next_batch(t_emb_pca, self.t_data, self.t_label, self.batch_size, shuffle=False):
                sample = i.shape[0]
                no  = np.random.laplace(size=(sample, self.pca_dim))
                feed_dict = {} 
                feed_dict[self.pca_input] = i
                feed_dict[self.data_p] = j
                feed_dict[self.label_p] = k.reshape(-1)
                feed_dict[self.pca_noise] =  no 
                _, _ = self.sess.run([self.opt_r_pca, self.opt_c_pca], feed_dict=feed_dict)

        mse_pca, mse_pca_lrr, mse_pca_krr = self.prediction_and_mse(t_emb_pca, v_emb_pca, self.v_label, PCA=True)
        acc_pca = self.prediction_and_accuracy(v_emb_pca, self.v_label, PCA=True)
        return acc, mse, mse_lrr, mse_krr, acc_pca, mse_pca, mse_pca_lrr, mse_pca_krr

    def get_emb(self, data):
        
        temp = []
        for i, j in self.next_batch(data, self.t_label, self.batch_size, shuffle=False):
                feed_dict = {} 
                feed_dict[self.data_p] = i
                #feed_dict[self.label_p] = j.reshape(-1)      
                temp.append(self.sess.run(self.compressing, feed_dict=feed_dict))

        return np.concatenate(temp, axis=0)

    def get_train_matrix(self):
        temp = []

        for i in self.t_data: 
            temp.append(i.reshape(1, -1))

        return np.concatenate(temp, axis=0)

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
                    yield np.array(t_data[i:]).reshape(-1, 64, 64, 3) , np.array(t_label[i:])
            else : 
                if MNIST_emb:
                    yield np.array(t_data[i: i+self.batch_size]) , np.array(t_label[i: i+self.batch_size])

                else :

                    yield np.array(t_data[i: i+self.batch_size]).reshape(-1, 64, 64, 3) , np.array(t_label[i: i+self.batch_size])



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
                yield np.array(t_emb[i:]), np.array(t_data[i:]).reshape(-1, 64, 64, 3) , np.array(t_label[i:])
            else : 
                yield np.array(t_emb[i: i+self.batch_size]), np.array(t_data[i: i+self.batch_size]).reshape(-1, 64, 64, 3) , np.array(t_label[i: i+self.batch_size])


    def prediction_and_accuracy(self, data, label, PCA=True):

        pred = [] 

        for i, j in self.next_batch(data, label, self.batch_size,  MNIST_emb=True):

            sample = i.shape[0]
            #no = np.random.multivariate_normal([0 for i in range(self.com_dim)], np.identity(self.com_dim), sample)
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

        TP = [] 
        NP = []
        for i, j in zip(self.v_label, predict):
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
            print("PCA with noise Accuracy (TPN): {}".format(acc))      

        else: 
            print("Deep features Accuracy: {}".format(accuracy))
            print("Deep features Accuracy (TPN): {}".format(acc))

        return accuracy

    def prediction_and_mse(self, train_data, data, label, PCA=True): 

        pred = []
        pred_lrr = []
        pred_krr = []
        for i, j in self.next_batch(data, label, self.batch_size, MNIST_emb=True):
            sample = i.shape[0]
            feed_dict = {} 
            if PCA : 
                no  = np.random.laplace(size=(sample, self.pca_dim))
                feed_dict[self.pca_input] = i
                feed_dict[self.pca_noise] = no
                #feed_dict[self.kernel_vec_p] = self.kernel_matrix(train_data, i, "rbf", 0.001, train=False)
                reco, reco_lrr, reco_krr = self.sess.run([self.upsampling_pca, self.upsampling_pca_lrr, self.upsampling_pca_krr], feed_dict=feed_dict)
            else :
                feed_dict[self.com_input_p] = i
                feed_dict[self.is_train] = False
                #feed_dict[self.kernel_vec_p] = self.kernel_matrix(train_data, i, "rbf", 0.001, train=False)
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
            error.append(mean_squared_error(self.plot(self.v_data[i]).flatten(), self.plot(predict[i]).flatten()))
            #error_lrr.append(mean_squared_error(self.plot(self.v_data[i]).flatten(), self.plot(predict_lrr[i]).flatten()))
            #error_krr.append(mean_squared_error(self.plot(self.v_data[i]).flatten(), self.plot(predict_krr[i]).flatten()))
            error_lrr.append(mean_squared_error(self.plot(self.v_data[i]).flatten(), self.plot(predict_lrr[i]).flatten()))
            error_krr.append(mean_squared_error(self.plot(self.v_data[i]).flatten(), self.plot(predict_krr[i]).flatten()))
        mse = np.mean(error)
        mse_lrr = np.mean(error_lrr)
        mse_krr= np.mean(error_krr)

        if PCA: 

            print("PCA with noise MSE (NN): {}".format(mse))
            print("PCA with noise MSE (LRR): {}".format(mse_lrr))
            print("PCA with noise MSE (KRR): {}".format(mse_krr))
            imsave("/home/pywu/bowei/Genki/reconstruction/pca_1.png", self.plot(predict[0]))
            imsave("/home/pywu/bowei/Genki/reconstruction/pca_2.png", self.plot(predict[1]))

        else : 
            print("Deep features MSE (NN): {}".format(mse))
            print("Deep features MSE (LRR): {}".format(mse_lrr))
            print("Deep features MSE (KRR): {}".format(mse_krr))

            imsave("/home/pywu/bowei/Genki/reconstruction/dnn_1.png", self.plot(predict[0]))
            imsave("/home/pywu/bowei/Genki/reconstruction/dnn_2.png", self.plot(predict[1]))

        return mse, mse_lrr, mse_krr

    def kernel_matrix(self, data_matrix, y, kernel_index, gamma, train=True):

        kernel_list = ['rbf', 'polynomial', 'laplacian', 'linear']

        if train : 

            if kernel_index =='rbf':
                K = rbf_kernel(data_matrix, gamma = gamma)          
            if kernel_index =='polynomial':
                K = polynomial_kernel(data_matrix, gamma = gamma)
            if kernel_index =='laplacian':
                K = laplacian_kernel(data_matrix, gamma = gamma)
            if kernel_index =='linear':
                #K = pairwise_kernels(data_matrix, 'linear')
                K = linear_kernel(data_matrix)
            return K 

        else :

            if kernel_index =='rbf':
                K = rbf_kernel(data_matrix, y, gamma = gamma)
            if kernel_index =='polynomial':
                K = polynomial_kernel(data_matrix, y, gamma = gamma)
            if kernel_index =='laplacian':
                K = laplacian_kernel(data_matrix, y, gamma = gamma)
            if kernel_index =='linear':
                #K = pairwise_kernels(data_matrix, y, 'linear')
                K = linear_kernel(data_matrix, y)
            return K
    '''
    def KRR_close_form(self, compressing_data, input_data):
        K = self.kernel_matrix(compressing_data, compressing_data, "rbf", 0.001, train=True)
        weights = np.dot(inv(K + 0.001 * np.identity(K.shape[0])), input_data)
        return weights
    '''

    
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

        rau = 1
        mu = np.mean(emb_matrix, axis=0)
        emb_matrix = emb_matrix - mu
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
        emb_matrix = emb_matrix - mu
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


    def inverse_transform(self, x):

        x = x+1 
        x = x/2 
        return x 

    def plot(self, x):
        #cifar_mean = np.array([0.4914, 0.4822, 0.4465])
        #cifar_std = np.array([0.2470, 0.2435, 0.2616])
        x = self.inverse_transform(x)
        #x = x - np.min(x)
        #x /= np.max(x)
        x *= 255  
        x= x.astype(np.uint8)
        x = x.reshape(64,64,3)
        return x 
