import numpy as np
import os 
import time
import random
import data as dataset
import matplotlib.pyplot as plt 
import pandas as pd 
from keras.datasets import cifar10
import math
import tensorflow as tf 
import tensorflow.contrib.layers as ly 
from scipy.misc import imrotate ,imread ,imsave,imresize
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
### For random fourier feature.
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from sklearn.utils import check_random_state
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import mean_squared_error
from numpy.linalg import pinv 


tf.set_random_seed(9)
np.random.seed(9)
plt.switch_backend('agg')

class CPGAN:
    def __init__(self,arg):
        self.arg = arg
        self.com_dim = self.arg.com_dim
        ### tunable parameter.
        self.gamma = self.arg.gamma
        self.mapping_dim = self.arg.mapping_dim
        self.seed = self.arg.seed

        self.num_epochs = 1800
        self.output_g = []
        self.output_c = []
        # ***********************************
        # For LRR and KRR close-form solution.
        # ***********************************
        self.t_data = self.get_train_matrix()
        self.train_set , self.val_set = self.load_data()
        
        self.build_model()

        print("The number of multiplication and addtion : {} and {}.".format(self.g_multiplication,self.g_addition))
        print("The number of multiplication and addtion : {} and {}.".format(self.c_multiplication,self.c_addition))

    def load_data(self):
        t_data,te_data,t_label,te_label = dataset.read_CIFAR10_subset()
        train_set = dataset.DataSet(t_data,t_label)
        test_set = dataset.DataSet(te_data,te_label)
        return train_set , test_set
        

    def _conv(self, input, filter_shape, stride):
        """Convolutional layer"""
        return tf.nn.conv2d(input,
                            filter=self.init_tensor(filter_shape),
                            strides=[1, stride, stride, 1],
                            padding="SAME")

    def _residual_unit(self, input_, in_filters, out_filters, stride, option=0):
        """
        Residual unit with 2 sub-layers
        When in_filters != out_filters:
        option 0: zero padding
        """
        # first convolution layer
        x = self.bo_batch_norm(input_,self.is_train)

        self.g_addition += 1 
        self.g_multiplication += 1

        x = tf.nn.relu(x)

        channel = x.get_shape().as_list()[-1]
        x = self._conv(x, [3, 3, in_filters, out_filters], stride)

        temp = x.get_shape().as_list()
        self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
        self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel) 

        # second convolution layer
        x = self.bo_batch_norm(x,self.is_train)

        self.g_addition += 1 
        self.g_multiplication += 1

        x = tf.nn.relu(x)
        channel = x.get_shape().as_list()[-1]
        x = self._conv(x, [3, 3, out_filters, out_filters], stride)

        self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
        self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel) 

        if in_filters != out_filters:
            if option == 0:
                difference = out_filters - in_filters
                left_pad = difference / 2
                right_pad = difference - left_pad
                identity = tf.pad(input_, [[0, 0], [0, 0], [0, 0], [left_pad, right_pad]])
                self.g_addition += 1 
                return x + identity
            else:
                print("Not implemented error")
                exit(1)
        else:
            self.g_addition += 1 
            return x + input_

    def init_tensor(self, shape):
        return tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=1.0))


    def adversary_lrr(self, latent, reuse=False):
        """
        The  close-form weights layer of Linear Ridge Regrssion.
        """
        with tf.variable_scope('adversary_lrr') as scope:  
            if reuse: 
                scope.reuse_variables()
            recontruction = ly.fully_connected(latent, 32*32*3, activation_fn=None, 
                                               weights_initializer=tf.contrib.layers.xavier_initializer(), 
                                               biases_initializer = None)
        return tf.reshape(recontruction, shape=[-1, 32, 32, 3])


    def adversary_krr(self, kernel_map, reuse=False):
        """
        The  close-form weights layer of Kernel. Ridge Regrssion (in the intrinsic space).
        """
        with tf.variable_scope('adversary_krr') as scope:  
            if reuse: 
                scope.reuse_variables()
            recontruction = ly.fully_connected(kernel_map, 32*32*3, activation_fn=None, 
                                               weights_initializer=tf.contrib.layers.xavier_initializer(), 
                                               biases_initializer = None)
        return tf.reshape(recontruction, shape=[-1, 32, 32, 3])


    def adversary_nn(self,latent,reuse=False):
        """
        Neural Network Adversaries.
        """
        with tf.variable_scope('adversary_nn') as scope:
            if reuse:
                scope.reuse_variables()
            dim = 32
            latent = ly.flatten(latent)
            latent = ly.fully_connected(latent, 4*4*64, activation_fn=tf.nn.relu)
            latent = self.bo_batch_norm(latent, self.is_train)
            latent = tf.reshape(latent, shape=[-1,4,4,64])
            upsample1 = ly.conv2d_transpose(latent, dim*4, kernel_size=3, stride=2, padding='SAME', 
                                            activation_fn=tf.nn.relu, 
                                            weights_initializer=tf.random_normal_initializer(0, 0.02))
            upsample2 = ly.conv2d_transpose(upsample1, dim*2, kernel_size=3, stride=2, padding='SAME', 
                                            activation_fn=tf.nn.relu, 
                                            weights_initializer=tf.random_normal_initializer(0, 0.02))
            upsample3 = ly.conv2d_transpose(upsample2, 3, kernel_size=3, stride=2, padding='SAME', 
                                            activation_fn=tf.nn.tanh, 
                                            weights_initializer=tf.random_normal_initializer(0, 0.02))
        return upsample3

    def RFF_map(self, input_tensor, seed, stddev, output_dim): 
        print("Information that the adversary is able to attain: {}".format(input_tensor))
        """
        Refer to the scikit learn package "RFF sampler" and tensorflow RFF mapping.
        """
        random_state = check_random_state(seed)
        #self._stddev = stddev
        gamma = stddev
        omega_matrix_shape = [3072, output_dim]
        bias_shape = [output_dim]

        """
        Tensorflow Version.

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
        """

        omega_matrix = constant_op.constant(np.sqrt(2 * gamma) *
           random_state.normal(size=omega_matrix_shape), dtype=dtypes.float32)

        bias = constant_op.constant(
            random_state.uniform(
            0.0, 2 * np.pi, size=bias_shape), dtype=dtypes.float32)

        x_omega_plus_bias = math_ops.add(
            math_ops.matmul(input_tensor, omega_matrix), bias)

        return math.sqrt(2.0 / output_dim) * math_ops.cos(x_omega_plus_bias)

    def build_model(self):

        ### Input placeholder
        self.image_p = tf.placeholder(tf.float32, shape=[None, 32,32,3])
        self.label_p = tf.placeholder(tf.int64, shape=[None, 10])
        self.is_train = tf.placeholder(tf.bool) ## For batchnormalization and dropout function
        self.learning_rate_p = tf.placeholder(tf.float32)
        self.noise_p = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        self.keep_prob = tf.placeholder(tf.float32)

        ### Privatizer
        self.latent = self.residual_g(self.image_p)

        ### Utility Classifier
        self.logit = self.utility_classifier(self.latent, 10)
        self.prob = tf.nn.softmax(self.logit)

        ### Adversary (Reconstructor)
        self.latent = ly.flatten(self.latent)
        ### Weights placeholder.
        self.lrr_mu_p = tf.placeholder(tf.float32, shape=[3072])
        self.krr_mu_p = tf.placeholder(tf.float32, shape=[self.mapping_dim])
        self.t_mu_p = tf.placeholder(tf.float32, shape=[32*32*3])
        self.krr_weights = tf.placeholder(tf.float32, shape=[self.mapping_dim, 32*32*3])
        self.lrr_weights = tf.placeholder(tf.float32, shape=[32*32*3, 32*32*3])
        self.lrr_mu = self.init_tensor([3072])
        self.krr_mu = self.init_tensor([self.mapping_dim])  
        self.t_mu = self.init_tensor([3072])

        ### *****Center-adjust the data or not.
        #self.latent_lrr = self.latent - self.lrr_mu 
        #self.latent_krr = self.latent - self.krr_mu
        self.kernel_map = self.RFF_map(self.latent, self.seed, self.gamma, self.mapping_dim)
        #self.kernel_map_deduct = self.kernel_map - self.krr_mu
        self.recon_nn = self.adversary_nn(self.latent) 
        self.recon_lrr = self.adversary_lrr(self.latent)
        self.recon_krr = self.adversary_krr(self.kernel_map)
        self.recon_lrr = self.recon_lrr # + tf.reshape(self.t_mu,[32,32,3])
        self.recon_krr = self.recon_krr # + tf.reshape(self.t_mu,[32,32,3])
        ### ***************************

        self.theta_r_nn = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='adversary_nn')
        self.theta_r_lrr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='adversary_lrr')
        self.theta_r_krr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='adversary_krr')
        self.theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='privatizer')
        self.theta_c_up = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='utility_classifier/decoder')
        self.theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='utility_classifier/classifier')

        ### Uitlity loss which follows the shake-shake regularization paper.
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob,1), tf.argmax(self.label_p,1)), tf.float32))
        l2_reg_loss = tf.add_n([tf.nn.l2_loss(var) for var in self.theta_c])
        utility_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label_p, logits=self.logit))
        self.loss_c = utility_loss + 0.0001 * l2_reg_loss

        ### Adversarial loss.
        self.loss_r_nn = tf.losses.mean_squared_error(self.image_p, self.recon_nn) 
        self.loss_r_lrr = tf.losses.mean_squared_error(self.image_p, self.recon_lrr) 
        self.loss_r_krr = tf.losses.mean_squared_error(self.image_p, self.recon_krr) 

        ### Combination loss.
        self.loss_g_nn = utility_loss - self.loss_r_nn
        self.loss_g_lrr = self.arg.trade_off*utility_loss - self.loss_r_lrr
        self.loss_g_krr = self.arg.trade_off*utility_loss - self.loss_r_krr

        print('The numbers of parameters in variable_scope G are : {}'.format(self.count_number_trainable_params(self.theta_g)))
        print('The numbers of parameters in variable_scope R are : {}'.format(self.count_number_trainable_params(self.theta_c)))

        ### ***************** Assign Operation *****************

        self.assign_op = []
        assign_lrr = self.theta_r_lrr[0].assign(self.lrr_weights)
        self.assign_op.append(assign_lrr)

        assign_krr = self.theta_r_krr[0].assign(self.krr_weights)
        self.assign_op.append(assign_krr)

        assign_t_mu = self.t_mu.assign(self.t_mu_p)
        self.assign_op.append(assign_t_mu)

        assign_lrr_mu = self.lrr_mu.assign(self.lrr_mu_p)
        self.assign_op.append(assign_lrr_mu)

        assign_krr_mu = self.krr_mu.assign(self.krr_mu_p)
        self.assign_op.append(assign_krr_mu)

        ###******************************************************

        uti_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(uti_update):
            ### Be careful to tune the learning of each three subnetworks.
            self.g_op_nn = tf.train.AdamOptimizer(0.001)
            self.g_opt_nn = self.g_op_nn.minimize(self.loss_g_nn, var_list=self.theta_g)

            self.g_op_lrr = tf.train.AdamOptimizer(0.001)
            self.g_opt_lrr = self.g_op_lrr.minimize(self.loss_g_lrr, var_list=self.theta_g)

            self.g_op_krr = tf.train.AdamOptimizer(0.001)
            self.g_opt_krr = self.g_op_krr.minimize(self.loss_g_krr, var_list=self.theta_g)

            ### small server's learning rate may be 0.001 or 0.01

            self.r_op = tf.train.AdamOptimizer(0.001)
            self.r_opt = self.r_op.minimize(self.loss_r_nn, var_list=self.theta_r_nn)

            self.c_op = tf.train.MomentumOptimizer(self.learning_rate_p, 0.9, use_nesterov=True)
            self.c_opt = self.c_op.minimize(self.loss_c, var_list = self.theta_c+self.theta_c_up)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=5)


    def bo_batch_norm(self,x, is_training, momentum=0.9, epsilon=0.00001):
        """
        Add a new batch-normalization layer.
        :param x: tf.Tensor, shape: (N, H, W, C).
        :param is_training: bool, train mode : True, test mode : False
        :return: tf.Tensor.
        """
        x = tf.layers.batch_normalization(x, momentum=momentum, epsilon=epsilon,training=is_training)
        return x

    def residual_g(self,image,reuse=False):

        stride = [1,1,1]
        filter_size = [3,3,3]

        self.g_addition = 0 
        self.g_multiplication = 0

        with tf.variable_scope('privatizer') as scope:
            if reuse : 
                scope.reuse_variables()
            channel = image.get_shape().as_list()[-1] 
            x = self._conv(image, [3, 3, 3, 3], 1)  
            temp = x.get_shape().as_list()

            self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
            self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel) 


            for i in range(len(filter_size)):
                for j in range(len([3,3,3])):
                    #with tf.variable_scope('unit_%d_sublayer_%d' % (i, j)):
                        if j == 0:
                            if i == 0:
                                # transition from init stage to the first stage stage
                                x = self._residual_unit(x, 3, filter_size[i], stride[i])
                            else:
                                x = self._residual_unit(x, filter_size[i - 1], filter_size[i], stride[i])
                        else:
                            x = self._residual_unit(x, filter_size[i], filter_size[i], stride[i])
            #print(x)
            return x 

    def generator_conv(self, image, reuse=False):
        ## Other choice of the privatizer.
        dim = 32
        with tf.variable_scope('compressor') as scope:
            if reuse : 
                scope.reuse_variables()
            conv1 = ly.conv2d(image,dim*1, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
            conv1 = self.bo_batch_norm(conv1,self.is_train)
            conv2 = ly.conv2d(conv1,dim*2, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
            conv2 = self.bo_batch_norm(conv2,self.is_train)
            conv3 = ly.conv2d(conv2,dim*4, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
            conv4 = self.bo_batch_norm(conv3, self.is_train)
            latent = ly.fully_connected(tf.reshape(conv4, shape=[-1,4*4*dim*4]), self.com_dim, activation_fn=tf.nn.relu)
        return latent 

    
    def utility_classifier(self,image,num_classes,reuse=False):
        """
        Build model.
        :param kwargs: dict, extra arguments for building ShakeNet.
            - batch_size: int, the batch size.
        :return d: dict, containing outputs on each layer.
        """

        batch_size = self.batch_size
        self.c_addition = 0 
        self.c_multiplication = 0

        with tf.variable_scope('utility_classifier/classifier'):
            if reuse:
                scope.reuse_variables()
            # input
            #X_input = self.X

            # first residual block's channels (26 2x32d --> 32)
            first_channel = 32 

            # the number of residual blocks (it means (depth-2)/6, i.e. 26 2x32d --> 4)
            num_blocks = 4

            # conv1 - batch_norm1
            with tf.variable_scope('conv1'):
                channel = image.get_shape().as_list()[-1]

                conv1 = self.conv_layer_no_bias(image, 3, 1, 16, padding='SAME')
                temp = conv1.get_shape().as_list()

                self.c_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel)
                self.c_addition += (temp[1]*temp[2]*temp[3]*((3*3*channel)-1))

                print('conv1.shape', conv1.get_shape().as_list())

            with tf.variable_scope('batch_norm1'):
                bnorm1 = self.batch_norm(conv1, is_training = self.is_train)
                self.c_multiplication += 1 
                self.c_addition += 1 
                print('batch_norm1.shape', bnorm1.get_shape().as_list())

            # shake stage 1
            with tf.variable_scope('shake_s1'):
                shake_s1 = self.shake_stage(bnorm1, first_channel, num_blocks, 1, batch_size)
                print('shake_s1.shape', shake_s1.get_shape().as_list())

            # shake stage 2
            with tf.variable_scope('shake_s2'):
                shake_s2 = self.shake_stage(shake_s1, first_channel * 2, num_blocks, 2, batch_size)
                print('shake_s2.shape', shake_s2.get_shape().as_list())

            # shake stage 3 with relu
            with tf.variable_scope('shake_s3'):
                shake_s3 = tf.nn.relu(self.shake_stage(shake_s2, first_channel * 4, num_blocks, 2, batch_size))
                print('shake_s3.shape', shake_s3.get_shape().as_list())
       
            avg_pool_shake_s3 = tf.reduce_mean(shake_s3, reduction_indices=[1, 2])
            print('avg_pool_shake_s3.shape', avg_pool_shake_s3.get_shape().as_list())

            # Flatten feature maps
            f_dim = int(np.prod(avg_pool_shake_s3.get_shape()[1:]))
            f_emb = tf.reshape(avg_pool_shake_s3, [-1, f_dim])
            print('f_emb.shape', f_emb.get_shape().as_list())


            x_shape = f_emb.get_shape().as_list()

            with tf.variable_scope('fc1'):
                logits = self.fc_layer(f_emb, num_classes)
                print('logits.shape', logits.get_shape().as_list())


            self.c_addition += (x_shape[-1]*10)
            self.c_multiplication += (x_shape[-1]*10)
            ### vary much scope that is used to avoid the name of each weight duplicate.
            # softmax
            #d['pred'] = tf.nn.softmax(d['logits'])
            return logits

    def shake_stage(self, x, output_filters, num_blocks, stride, batch_size):
        """
        Build sub stage with many shake blocks.
        :param x: tf.Tensor, input of shake_stage, shape: (N, H, W, C).
        :param output_filters: int, the number of output filters in shake_stage.
        :param num_blocks: int, the number of shake_blocks in one shake_stage.
        :param stride: int, the stride of the sliding window to be applied shake_block's branch. 
        :param batch_size: int, the batch size.
        :param d: dict, the dictionary for saving outputs of each layers.
        :return tf.Tensor.
        """
        shake_stage_idx = int(math.log2(output_filters // 16))  #FIXME if you change 'first_channel' parameter

        for block_idx in range(num_blocks):
            stride_block = stride if (block_idx == 0) else 1
            with tf.variable_scope('shake_s{}_b{}'.format(shake_stage_idx, block_idx)):
                x = self.shake_block(x, shake_stage_idx, block_idx, output_filters, stride_block, batch_size)
            #d['shake_s{}_b{}'.format(shake_stage_idx, block_idx)] = x
        return x


    def shake_block(self, x, shake_stage_idx, block_idx, output_filters, stride, batch_size):
        """
        Build one shake-shake blocks with branch and skip connection.
        :param x: tf.Tensor, input of shake_block, shape: (N, H, W, C).
        :param shake_layer_idx: int, the index of shake_stage.
        :param block_idx: int, the index of shake_block.
        :param output_filters: int, the number of output filters in shake_block.
        :param stride: int, the stride of the sliding window to be applied shake_block's branch. 
        :param batch_size: int, the batch size.
        :return tf.Tensor.
        """
        num_branches = 2
        # Generate random numbers for scaling the branches.
        
        rand_forward = [
          tf.random_uniform([batch_size, 1, 1, 1], minval=0, maxval=1, dtype=tf.float32) for _ in range(num_branches)
        ]
        rand_backward = [
          tf.random_uniform([batch_size, 1, 1, 1], minval=0, maxval=1, dtype=tf.float32) for _ in range(num_branches)
        ]

        #prob_l = rand_backward.get_shape().as_list()

        #self.c_addition += (len(prob_l)-1).  not include in the network !
        #self.c_addition += (len(prob_l)-1).  not include in the network !

        # Normalize so that all sum to 1.
        total_forward = tf.add_n(rand_forward)
        total_backward = tf.add_n(rand_backward)
        rand_forward = [samp / total_forward for samp in rand_forward]
        rand_backward = [samp / total_backward for samp in rand_backward]
        zipped_rand = zip(rand_forward, rand_backward)

        branches = []
        for branch, (r_forward, r_backward) in enumerate(zipped_rand):

            with tf.variable_scope('shake_s{}_b{}_branch_{}'.format(shake_stage_idx, block_idx, branch)):
                b = self.shake_branch(x, output_filters, stride, r_forward, r_backward, num_branches)
                branches.append(b)

        res = self.shake_skip_connection(x, output_filters, stride)

        self.c_addition += (len(branches)-1)
        self.c_addition += 1 
        return res + tf.add_n(branches)

    def shake_branch(self, x, output_filters, stride, random_forward, random_backward, num_branches):
        """
        Build one shake-shake branch.
        :param x: tf.Tensor, input of shake_branch, shape: (N, H, W, C).
        :param output_filters: int, the number of output filters in shake_branch.
        :param stride: int, the stride of the sliding window to be applied shake_block's branch. 
        :param random_forward: tf.float32, random scalar weight, in paper (alpha or 1 - alpha) for forward propagation.
        :param random_backward: tf.float32, random scalar weight, in paper (alpha or 1 - alpha) for backward propagation.
        :param num_branches: int, the number of branches.
        :return tf.Tensor.
        """
        # relu1 - conv1 - batch_norm1 with stride = stride
        with tf.variable_scope('branch_conv_bn1'):
           x = tf.nn.relu(x) 

           channel = x.get_shape().as_list()[-1]

           x = self.conv_layer_no_bias(x, 3, stride, output_filters)

           temp = x.get_shape().as_list()
           self.c_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel)
           self.c_addition += (temp[1]*temp[2]*temp[3]*((3*3*channel-1)))

           x = self.batch_norm(x, is_training=self.is_train) 

           self.c_multiplication +=1 
           self.c_addition += 1 

        # relu2 - conv2 - batch_norm2 with stride = 1
        with tf.variable_scope('branch_conv_bn2'):
           x = tf.nn.relu(x)

           channel = x.get_shape().as_list()[-1]

           x = self.conv_layer_no_bias(x, 3, 1, output_filters) # stirde = 1

           temp = x.get_shape().as_list()
           self.c_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel)
           self.c_addition += (temp[1]*temp[2]*temp[3]*((3*3*channel-1)))

           x = self.batch_norm(x, is_training=self.is_train)

           self.c_multiplication +=1 
           self.c_addition += 1 

        # condition on the forward and backward need different flow chart.
        # refer to https://stackoverflow.com/questions/36456436/how-can-i-define-only-the-gradient-for-a-tensorflow-subgraph/36480182#36480182
        x = tf.cond(self.is_train, lambda: x * random_backward + tf.stop_gradient(x * random_forward - x * random_backward), 
                    lambda: x / num_branches)

        self.c_multiplication += 1 
        return x


    def shake_skip_connection(self, x, output_filters, stride):
        """
        Build one shake-shake skip connection.
        :param x: tf.Tensor, input of shake_branch, shape: (N, H, W, C).
        :param output_filters: int, the number of output filters in shake_branch.
        :param stride: int, the stride of the sliding window to be applied shake_block's branch. 
        :return tf.Tensor.
        """
        input_filters = int(x.get_shape()[-1])
        
        if input_filters == output_filters:
           return x

        x = tf.nn.relu(x)

        # Skip connection path 1.
        # avg_pool1 - conv1 
        with tf.variable_scope('skip1'):
            path1 = tf.nn.avg_pool(x, [1, 1, 1, 1], [1, stride, stride, 1], "VALID")

            channel = path1.get_shape().as_list()[-1]

            path1 = self.conv_layer_no_bias(path1, 1, 1, int(output_filters / 2))

            temp = path1.get_shape().as_list()
            self.c_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel)
            self.c_addition += (temp[1]*temp[2]*temp[3]*((3*3*channel)-1))

        # Skip connection path 2.
        # pixel shift2 - avg_pool2 - conv2 
        with tf.variable_scope('skip2'):
            path2 = tf.pad(x, [[0, 0], [0, 1], [0, 1], [0, 0]])[:, 1:, 1:, :]
            path2 = tf.nn.avg_pool(path2, [1, 1, 1, 1], [1, stride, stride, 1], "VALID")

            channel = path2.get_shape().as_list()[-1]
            path2 = self.conv_layer_no_bias(path2, 1, 1, int(output_filters / 2))
            temp = path2.get_shape().as_list()

            self.c_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel)
            self.c_addition += (temp[1]*temp[2]*temp[3]*((3*3*channel)-1))

        # Concatenation path 1 and path 2 and apply batch_norm
        with tf.variable_scope('concat'):
            concat_path = tf.concat(values=[path1, path2], axis= -1)
            bn_path = self.batch_norm(concat_path, is_training=self.is_train)
            
            self.c_addition +=1 
            self.c_multiplication +=1 

        return bn_path


    def weight_variable(self,shape):
        """
        Initialize a weight variable with given shape,
        by Xavier initialization.
        :param shape: list(int).
        :return weights: tf.Variable.
        """
        tf.set_random_seed(9)
        np.random.seed(9)
        weights = tf.get_variable('weights', shape, tf.float32, tf.contrib.layers.xavier_initializer())

        return weights

    def bias_variable(self, shape, value=1.0):
        """
        Initialize a bias variable with given shape,
        with given constant value.
        :param shape: list(int).
        :param value: float, initial value for biases.
        :return biases: tf.Variable.
        """
        tf.set_random_seed(9)
        np.random.seed(9)
        biases = tf.get_variable('biases', shape, tf.float32,
                                 tf.constant_initializer(value=value))
        return biases

    def conv2d(self,x, W, stride, padding='SAME'):
        """
        Compute a 2D convolution from given input and filter weights.
        :param x: tf.Tensor, shape: (N, H, W, C).
        :param W: tf.Tensor, shape: (fh, fw, ic, oc).
        :param stride: int, the stride of the sliding window for each dimension.
        :param padding: str, either 'SAME' or 'VALID',
                             the type of padding algorithm to use.
        :return: tf.Tensor.
        """
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

    def max_pool(self,x, side_l, stride, padding='SAME'):
        """
        Performs max pooling on given input.
        :param x: tf.Tensor, shape: (N, H, W, C).
        :param side_l: int, the side length of the pooling window for each dimension.
        :param stride: int, the stride of the sliding window for each dimension.
        :param padding: str, either 'SAME' or 'VALID',
                             the type of padding algorithm to use.
        :return: tf.Tensor.
        """
        return tf.nn.max_pool(x, ksize=[1, side_l, side_l, 1],
                              strides=[1, stride, stride, 1], padding=padding)

    def conv_layer_no_bias(self,x, side_l, stride, out_depth, padding='SAME'):
        """
        Add a new convolutional layer.
        :param x: tf.Tensor, shape: (N, H, W, C).
        :param side_l: int, the side length of the filters for each dimension.
        :param stride: int, the stride of the filters for each dimension.
        :param out_depth: int, the total number of filters to be applied.
        :param padding: str, either 'SAME' or 'VALID',
                             the type of padding algorithm to use.
        :return: tf.Tensor.
        """
        tf.set_random_seed(9)
        np.random.seed(9)
        in_depth = int(x.get_shape()[-1])

        filters = self.weight_variable([side_l, side_l, in_depth, out_depth])
          
        return self.conv2d(x, filters, stride, padding=padding)

    def fc_layer(self,x, out_dim, **kwargs):
        tf.set_random_seed(9)
        np.random.seed(9)
        in_dim = int(x.get_shape()[-1])

        weights = self.weight_variable([in_dim, out_dim])
        biases = self.bias_variable([out_dim], value=0.1)
        return tf.matmul(x, weights) + biases

    def batch_norm(self,x, is_training, momentum=0.9, epsilon=0.00001):
        tf.set_random_seed(9)
        #np.random.seed(9)
        x = tf.layers.batch_normalization(x, momentum=momentum, epsilon=epsilon ,training=is_training)
        return x

    # ****************************
    # These functions is no longer used in this scipt after authors find another source code to deal with the data augumentation.
    # ****************************
    def batch_random_rotate_image(self,image):
        angle = np.random.uniform(low=-5.0, high=5.0)
        a = []
        cifar_mean = np.array([0.4914,0.4822,0.4465])
        cifar_std = np.array([0.2470,0.2435,0.2616])
        for i in image:
            a.append(imrotate(i, angle, 'bicubic'))

        for i in range(len(a)):
            #a[i] = a[i]/255
            a[i] = (a[i]/127.5)-1
            #a[i] -= cifar_mean
            #a[i] /= cifar_std
        return a

    def batch_mirror_image(self,image):
        a = []
        for i in image :
            a.append(np.flipud(i))
        return a

    def batch_crop_image(self,image):
        a = [] 
        for img in image : 
            reflection = bool(np.random.randint(2))
            if reflection :
                img = np.fliplr(img)

            image_pad = np.pad(img,((4,4),(4,4),(0,0)),mode='constant')

            crop_x1 = random.randint(0,8)
            crop_x2 = crop_x1 + 32

            crop_y1 = random.randint(0,8)
            crop_y2 = crop_y1 + 32

            image_crop = image_pad[crop_x1:crop_x2,crop_y1:crop_y2]
            a.append(image_crop)

        return a
    # end here ****************************

    def plot_10slot(self,name):
        j = np.array(self.te_data[:128])
        k = np.array([i for i in range(128)])
        no = np.random.normal(size=(128,32,32,3))
        uu = self.sess.run(self.latent, feed_dict={self.image_p:j.reshape(128,32,32,3), self.label_p:k, self.is_train:False})
        yy = self.sess.run(self.up, feed_dict={ self.latent:uu})
        plt.figure(figsize=(10, 2))
        n = 10
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(self.plot(self.te_data[i]))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # display reconstruction

            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(self.plot(yy[i]))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig('reconstructed_image'+str(name))

    def plot(self,x):
        #cifar_mean = np.array([0.4914, 0.4822, 0.4465])
        #cifar_std = np.array([0.2470, 0.2435, 0.2616])
        x = x - np.min(x)
        x /= np.max(x)
        x *= 255  
        x= x.astype(np.uint8)
        x = x.reshape(32,32,3)
        return x 

    def compute_acc(self, te_data, te_label, is_train=False):

        acc_list = []

        for j , k in self.next_batch(te_data,te_label):
            b = k.shape[0]
            no = np.random.normal(size=(b,32,32,3))
            pred = self.sess.run(self.prob,feed_dict={self.image_p:j.reshape(b,32,32,3),self.label_p:k.reshape(-1),self.is_train:False})
            acc_list.append(pred)

        if is_train :
            preds = np.concatenate((acc_list),axis=0)
            preds = preds[0:50000]          
        else :
            preds = np.concatenate((acc_list),axis=0)
            preds = preds[0:10000]

        ac = accuracy_score(np.argmax(preds,axis=1),te_label)
        return ac

    def predict(self, data):
        if data.labels is not None : 
            assert len(data.labels.shape) > 1 , 'Labels must be one-hot encoded'
        num_classes = int(data.labels.shape[-1])
        pred_size = data.num_examples
        num_steps = pred_size // self.batch_size
        _y_pred = []
        for i in range(num_steps+1):
            if i == num_steps:
                _batch_size = pred_size - num_steps * self.batch_size
            else : 
                _batch_size = self.batch_size

            no = np.random.laplace(size=(_batch_size,32,32,3))
            X , _ = data.next_batch(_batch_size,shuffle=False,augment=False,is_train=False)
            y_pred = self.sess.run(self.prob,feed_dict={self.image_p:X, self.is_train:False,self.noise_p:no})
            _y_pred.append(y_pred)

        _y_pred = np.concatenate(_y_pred, axis=0)
        y_true = data.labels
        acc = accuracy_score(np.argmax(y_true,1),np.argmax(_y_pred,1))
        return acc


    def compute_reco_mse(self, data):

        ##### after assign all the weights !!!!! 

        error_nn = []
        error_lrr = []
        error_krr = []

        num_classes = int(data.labels.shape[-1])
        pred_size = data.num_examples
        num_steps = pred_size // self.batch_size

        for i in range(num_steps+1):
            if i == num_steps:
                _batch_size = pred_size - num_steps * self.batch_size
            else : 
                _batch_size = self.batch_size
            no = np.random.laplace(size=(_batch_size,32,32,3))   
            X , _ = data.next_batch(_batch_size, shuffle=False, augment=False, is_train=False)


            up_nn = self.sess.run(self.recon_nn, feed_dict={self.image_p:X, self.is_train:False, self.noise_p:no})
            up_lrr = self.sess.run(self.recon_lrr, feed_dict={self.image_p:X, self.noise_p:no, self.is_train:False})
            up_krr = self.sess.run(self.recon_krr, feed_dict={self.image_p:X, self.noise_p:no, self.is_train:False})

            for k in range(len(up_nn)):
                error_nn.append(mean_squared_error(X[k].flatten(), up_nn[k].flatten())) 
                error_lrr.append(mean_squared_error(X[k].flatten(), up_lrr[k].flatten()))#+train_mu)) 
                error_krr.append(mean_squared_error(X[k].flatten(), up_krr[k].flatten()))#+train_mu)) 


        #imsave('cpgan_log/nn_reco.png', self.plot(up_nn[0]))
        #imsave('cpgan_log/lrr_reco.png', self.plot(up_lrr[0]))#+train_mu))
        #imsave('cpgan_log/krr_reco.png', self.plot(up_krr[0]))#+train_mu))
        #print('Average MSE among all testing images is {}'.format(np.mean(error)))

        return np.mean(error_nn), np.mean(error_lrr), np.mean(error_krr)

    def KRR_close_form(self, emb_matrix, train_matrix, train_mu):
        # Use the random fourier transform to approximate the RBF kernel 
        # Note that the training data is too large so that we use the intrinsic space mapping 
        # And use the tensorflow conrtrib package to get the RFF mapping rather than hand crafting
        # More information refers to https://github.com/hichamjanati/srf  
        emb_list = []
        real_list = [] 
        rau = 0.00001
        mu = np.mean(emb_matrix, axis=0)
        #emb_matrix = emb_matrix - mu 
        emb_matrix = emb_matrix.T 
        s = np.dot(emb_matrix, emb_matrix.T)
        a,b = s.shape
        identity = np.identity(a)
        s_inv = np.linalg.inv(s+ rau * np.identity(a))
        #train_norm = train_matrix - train_mu
        weights = np.dot(np.dot(s_inv,emb_matrix), train_matrix)
        print('Shape of the KRR weight: {}'.format(weights.shape))
        return weights, mu


    def LRR_close_form(self, emb_matrix, train_matrix, train_mu):
        emb_list = []
        real_list = []
        count = 0 
        mu = np.mean(emb_matrix, axis=0)
        #emb_matrix = emb_matrix - mu 
        emb_matrix = emb_matrix.T
        rau = 0.00001
        s = np.dot(emb_matrix, emb_matrix.T)
        h,w = s.shape
        s_inv = np.linalg.inv(s+ rau*np.identity(h))
        #train_norm = train_matrix - train_mu
        weights = np.dot(np.dot(s_inv, emb_matrix), train_matrix)
        print('Shape of the LRR weight: {}'.format(weights.shape))
        return weights, mu

    def get_emb_matrix(self):
        train_size = self.train_set.num_examples
        num_steps = train_size // self.batch_size
        count = 0
        for i in range(num_steps+1):
            if i == num_steps:
                _batch_size = train_size - num_steps * self.batch_size
            else : 
                _batch_size = self.batch_size
            no = np.random.laplace(size=(self.batch_size, 32, 32, 3))
            X , y_true = self.train_set.next_batch(_batch_size, shuffle=False, augment=False, is_train=False)
            feed_dict = {self.image_p:X, self.label_p:y_true, self.is_train:False, self.noise_p:no}    
            compressing_representation, kernel_mapping = self.sess.run([self.latent, self.kernel_map], feed_dict=feed_dict)

            if count == 0 : 
                emb_matrix_lrr = compressing_representation
                emb_matrix_krr = kernel_mapping
                count+=1 
            else : 
                emb_matrix_lrr = np.concatenate((emb_matrix_lrr, compressing_representation), axis=0)
                emb_matrix_krr = np.concatenate((emb_matrix_krr, kernel_mapping), axis=0)
                count+=1     
        return emb_matrix_lrr, emb_matrix_krr 

    def get_train_matrix(self): 
        train_size = self.train_set.num_examples
        num_steps = train_size // self.batch_size
        count = 0
        for i in range(num_steps+1):
            if i == num_steps:
                _batch_size = train_size - num_steps * self.batch_size
            else : 
                _batch_size = self.batch_size

            no = np.random.laplace(size=(self.batch_size, 32, 32, 3))
            X , y_true = self.train_set.next_batch(_batch_size, shuffle=False, augment=False, is_train=False)
            if count == 0 : 
                train_matrix = X.reshape(-1, 32*32*3)
                count+=1 
            else : 
                train_matrix = np.concatenate((train_matrix, X.reshape(-1, 32*32*3)), axis=0)
                count+=1 

        print('Successfully get flatted train matrix !!!!')
        return train_matrix

    def sklearn_sol(self, train_matrix, val_matrix, emb_matrix, emb_matrix_te, gamma ,mapping_dim, seed): 

        rbf_feature = RBFSampler(gamma=gamma, n_components=mapping_dim, random_state=seed)
        emb_matrix = rbf_feature.fit_transform(emb_matrix.reshape(-1, 32*32*3))
        #rau = self.arg.rau
        rau = 0.0001
        #emb_matrix = emb_matrix[:len(self.t_data)]
        mu = np.mean(emb_matrix, axis=0)
        emb_matrix_1 = emb_matrix #- mu
        #emb_matrix_1 = emb_matrix
        emb_matrix = emb_matrix_1.T 
        #print(np.mean(emb_matrix))
        s = np.dot(emb_matrix, emb_matrix.T)
        a,b = s.shape
        identity = np.identity(a)
        s_inv = np.linalg.inv(s + rau * np.identity(a))

        output_mu = np.mean(train_matrix, axis=0)
        output_norm = train_matrix# - output_mu
        weights = np.dot(np.dot(s_inv, emb_matrix), output_norm)
        #weights = np.dot(np.dot(s_inv, emb_matrix), self.t_label)
        pred = np.dot(emb_matrix_1, weights) # + output_mu       
        emb_matrix_te = rbf_feature.fit_transform(emb_matrix_te.reshape(-1, 3072))
        #emb_matrix -= mu

        pred = np.dot(emb_matrix_te, weights) #+ output_mu        
        #print(pred.shape)
        mse_trace = []
        for i in range(len(val_matrix)):
            mse_trace.append(mean_squared_error(val_matrix[i].flatten(), pred[i]))
        
        return np.mean(mse_trace)


    def tune_kernel_matrix(self, train_matrix, emb_matrix):
        # Choose the best kernel parameter by tuning gamma, mapping dimension and changing the seed..
        # This is time-consuming so that we not recommend you to use it!
        train_size = self.train_set.num_examples
        num_steps = train_size // self.batch_size
        count = 0

        data = self.val_set

        if data.labels is not None : 
            assert len(data.labels.shape) > 1 , 'Labels must be one-hot encoded'
        num_classes = int(data.labels.shape[-1])
        pred_size = data.num_examples
        num_steps = pred_size // self.batch_size
        _y_pred = []

        for i in range(num_steps+1):
            if i == num_steps:
                _batch_size = pred_size - num_steps * self.batch_size
            else : 
                _batch_size = self.batch_size
            no = np.random.laplace(size=(_batch_size,32,32,3))   
            X , _ = data.next_batch(_batch_size, shuffle=False, augment=False, is_train=False)
            feed_dict = {self.image_p:X, self.is_train:False, self.noise_p:no}    
            uu = self.sess.run(self.latent, feed_dict={self.image_p:X, self.keep_prob:1, self.is_train:False})

            if count == 0 : 
                emb_matrix_te = uu.reshape(-1, 32*32*3)
                val_matrix = X.reshape(-1, 32*32*3)
                count+=1 
            else : 
                emb_matrix_te = np.concatenate((emb_matrix_te,  uu.reshape(-1, 32*32*3)), axis=0)
                val_matrix = np.concatenate((val_matrix,  X.reshape(-1, 32*32*3)), axis=0)

                count+=1 


        gamma_record = [] 
        seed_record = []
        dimension_record = [] 
        mse_record = []

        mse = self.sklearn_sol(train_matrix, val_matrix, emb_matrix, emb_matrix_te, self.gamma, self.mapping_dim, self.seed)
        print("First MSE that sklean solution caused: {}.".format(mse))

        gamma_record.append(self.gamma)
        seed_record.append(self.seed)
        dimension_record.append(self.mapping_dim)
        mse_record.append(mse)

        gamma_choices = list(np.arange(0, 100, 0.000001))
        seed_choices = [i for i in range(100)]
        dimension_choices = [(i+500) for i in range(0,8500,500)]

        for i in range(10):

            self.gamma = random.sample(gamma_choices, 1)[0]
            self.seed = random.sample(seed_choices, 1)[0]
            self.mapping_dim = random.sample(dimension_choices, 1)[0]

            gamma_record.append(self.gamma)
            seed_record.append(self.seed)
            dimension_record.append(self.mapping_dim)
            mse = self.sklearn_sol(train_matrix, val_matrix, emb_matrix, emb_matrix_te, self.gamma, self.mapping_dim, self.seed)
            mse_record.append(mse)


            print("Mse = {}. Parameters: is gamma, seed, mapping_dim:{:.3f}, {:.3f}, {:.3f}.".
                format(mse, self.gamma, self.seed, self.mapping_dim))

        index = np.argmin(mse_record)

        # back to the RFF mapping and train the network with this new mapping dimension.

        # self.gamma = gamma_record[index]
        # self.seed = seed_record[index]
        # self.mapping_dim = dimension_record[index]

        gamma = gamma_record[index]
        seed = seed_record[index]
        mapping_dim = dimension_record[index]

        return gamma, seed, mapping_dim


    def assign(self, train_matrix, train_mu, epo):

        feed_dict_assign = {}
        tf.reset_default_graph()
        self.build_model()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=5)

        if epo != 0 : 
            temp_c = [] 
            c_file = np.load('multi_adv/theta_c.npy')
            for i in range(len(c_file)): 
                temp_c.append(tf.assign(self.theta_c[i], c_file[i]))

            #temp_g = [] 
            #g_file = np.load('multi_adv/theta_g.npy')
            #for i in range(len(c_file)): 
            #    temp_g.append(tf.assign(self.theta_g[i], g_file[i]))
            temp_g = self.load_g()
  
            temp_r = [] 
            r_file = np.load('multi_adv/theta_r.npy')
            for i in range(len(r_file)): 
                temp_r.append(tf.assign(self.theta_r_nn[i], r_file[i]))


            self.sess.run([temp_c, temp_g])


        if epo == 0 : 
            #np.save('multi_adv/theta_g.npy', self.sess.run(self.theta_g))
            self.save_g()

        emb_matrix_lrr, emb_matrix_krr = self.get_emb_matrix()

        start = time.time()
        gamma, seed, mapping_dim = self.tune(train_matrix, emb_matrix_lrr)
        end = time.time()

        print('Time for tuning the RFF parameters: {}.'.format(end-start))

        #### *****************  For the empirical space computation, but doesn;t have enought GPU memory !!
        #mse, emp_weights = self.empirical_sol(emb_matrix_lrr, train_matrix)
        #print('Average MSE among all testing images is {}.(KRR (empirical space)).'.format(mse))
        #feed_dict[self.emp_weights] = emp_weights
        #### ***************** 

        start = time.time()
        train_data = []
        train_label = []
        lr = []

        train_size = self.train_set.num_examples
        num_steps_per_epoch = train_size // self.batch_size
        num_steps = self.num_epochs * num_steps_per_epoch
        citers = 25


        if epo == 0: 
            self.count = 0 


        for _ in range(num_steps_per_epoch):
            #no = np.random.normal(size=(self.batch_size,32,32,3))
            no = np.random.laplace(size=(self.batch_size, 32, 32, 3))
            X , y_true = self.train_set.next_batch(self.batch_size, shuffle=True, augment=True, is_train=True)

            train_data.append(X)
            train_label.append(y_true)
            lr.append(self.curr_learning_rate)
            feed_dict = {self.image_p:X, self.label_p:y_true, self.is_train:True, self.learning_rate_p:self.curr_learning_rate, self.noise_p:no}
            for _ in range(citers):
                _ = self.sess.run([self.r_opt],feed_dict=feed_dict)
            _, loss, y_pred = self.sess.run([self.c_opt,self.loss_c,self.prob],feed_dict=feed_dict)
            self._update_learning_rate_cosine(self.count, num_steps)
            self.count +=1
        end = time.time()  

        np.save('multi_adv/theta_c.npy', self.sess.run(self.theta_c))
        np.save('multi_adv/theta_r.npy', self.sess.run(self.theta_r_nn))


        tf.reset_default_graph()
        self.gamma = gamma
        self.seed = seed
        self.mapping_dim = mapping_dim
        self.build_model()

        #self.assign_op += self.assign_each_part()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=5)

        #if epo != 0 : 
        temp_c = [] 
        c_file = np.load('multi_adv/theta_c.npy')
        for i in range(len(c_file)): 
            temp_c.append(tf.assign(self.theta_c[i], c_file[i]))

        #temp_g = [] 
        #g_file = np.load('multi_adv/theta_g.npy')
        #for i in range(len(g_file)): 
        #   temp_g.append(tf.assign(self.theta_g[i], g_file[i]))
        temp_g = self.load_g()


        temp_r = [] 
        r_file = np.load('multi_adv/theta_r.npy')
        for i in range(len(r_file)): 
            temp_r.append(tf.assign(self.theta_r_nn[i], r_file[i]))

        self.sess.run([temp_c, temp_g, temp_r])

        emb_matrix_lrr, emb_matrix_krr = self.get_emb_matrix()

        #np.save('multi_adv/theta_g.npy', self.sess.run(theta_g))

        #self.build_model()
        #self.assign_op += self.assign_each_part()
        #self.sess = tf.Session()
        #self.sess.run(tf.global_variables_initializer())
        #self.saver = tf.train.Saver(max_to_keep=5)

        error_list = []
        update_choice = [self.g_opt_nn, self.g_opt_lrr, self.g_opt_krr]
        assign_op = []

        lrr_weights, lrr_mu = self.LRR_close_form(emb_matrix_lrr, train_matrix, train_mu)
        #assign_op.append(tf.assign(self.theta_r_lrr[0], lrr_weights))
        #print(assign_op)
        feed_dict_assign[self.lrr_mu_p] = lrr_mu
        feed_dict_assign[self.lrr_weights] = lrr_weights

        krr_weights, krr_mu = self.KRR_close_form(emb_matrix_krr, train_matrix, train_mu)

        feed_dict_assign[self.krr_mu_p] = krr_mu
        feed_dict_assign[self.krr_weights] = krr_weights
        feed_dict_assign[self.t_mu_p] = train_mu

        #assign_op.append(tf.assign(self.theta_r_krr[0], krr_weights))

        #assign_op.append(tf.assign(self.lrr_mu, lrr_mu))
        #assign_op.append(tf.assign(self.krr_mu, krr_mu))
        #assign_op.append(tf.assign(self.t_mu, train_mu))

        #self.sess.run(assign_op)

        self.sess.run(self.assign_op, feed_dict = feed_dict_assign)

        error_nn, error_lrr, error_krr = self.compute_reco_mse(self.val_set)
        error_list.append(error_nn) 
        error_list.append(error_lrr)
        error_list.append(error_krr)
        print('Average MSE among all testing images is {}, {}, {}.(nn,lrr,krr)'.format(error_nn, error_lrr, error_krr))
        optimize_g = update_choice[np.argmin(error_list)]
        return optimize_g, feed_dict_assign, train_data, train_label, lr


    def train(self):
        ### our server setting is 5 for citer.  
        gen = 1

        train_size = self.train_set.num_examples
        num_steps_per_epoch = train_size // self.batch_size
        num_steps = self.num_epochs * num_steps_per_epoch
        self.curr_learning_rate = self.init_learning_rate

        count = 0
        update_choice = [self.g_opt_nn, self.g_opt_lrr, self.g_opt_krr]
        train_matrix = self.get_train_matrix()
        train_mu = np.mean(train_matrix, axis=0)
        temp = train_matrix-train_mu
        print("training matrix ceter adjust : {}".format(np.mean(temp)))

        for epo in range(self.num_epochs):

            start = time.time()
            optimize_g, feed_dict, train_data, train_label, lr = self.assign(train_matrix, train_mu, epo)
            end = time.time()
            print("Training for R and C costs about {}.".format(end-start))

            #for _ in range(num_steps_per_epoch):
            start = time.time()

            for i in range(len(train_data)):
                no = np.random.laplace(size=(self.batch_size, 32, 32, 3))
                feed_dict = {}
                feed_dict[self.image_p] = train_da
                feed_dict = {self.image_p: train_data[i], self.label_p: train_label[i], self.is_train:True, self.learning_rate_p:lr[i], self.noise_p:no}
                for _ in range(gen):

                    _ = self.sess.run([optimize_g], feed_dict=feed_dict)

            #self.save_g()

            end = time.time()
            print("Training for G costs about {}.".format(end-start))
            av_acc = self.predict(self.val_set)        
            if epo % 30 == 0 : 
                self.save_g()
                self.saver.save(self.sess, "model_cifar10")


    def _update_learning_rate_cosine(self, global_step, num_iterations):
        """
        update current learning rate, using Cosine function without restart(Loshchilov & Hutter, 2016).
        """ 
        global_step = min(global_step, num_iterations)
        decay_step = num_iterations
        alpha = 0
        cosine_decay = 0.5 * (1 + math.cos(math.pi * global_step / decay_step))
        decayed = (1 - alpha) * cosine_decay + alpha
        new_learning_rate = self.init_learning_rate * decayed
        #self.c_op._lr = new_learning_rate
        self.curr_learning_rate = new_learning_rate  


    def save_g(self):
        
        temp = self.sess.run(self.theta_g)
        a = []
        for i in range(len(temp)):
            a.append(temp[i].flatten())
        np.save('multi_adv/theta_g.npy',a)

    def load_g(self):
        temp = np.load('multi_adv/theta_g.npy')
        assign = []
        for i in range(len(self.theta_g)):
            if temp[i].shape[0] > 3 :
                assign.append(tf.assign(self.theta_g[i],temp[i].reshape(3,3,3,3)))
            else : 
                assign.append(tf.assign(self.theta_g[i],temp[i]))

        return assign

    def test(self):
        #self.saver.restore(self.sess,'cpgan_log/model_137')
        self.saver.restore(self.sess,self.arg.model_path)
        print('successfully restore')
        #print(len(self.sess.run(self.theta_g)))
        av_acc = self.predict(self.val_set)
        print('Testing accuracy is {}.'.format(av_acc))
        #temp = self.sess.run(self.theta_g)
        '''
        a = []
        for i in range(len(temp)):
            a.append(temp[i].flatten())
        for i in range(len(a)):
            print(a[i].shape)
        '''
        #self.compute_acc(self.te_data,self.te_label)
        #self.plot_10slot('restore')  
        #np.save('Male_2_noise_loop30/g.npy',self.sess.run(self.theta_g))

    def count_number_trainable_params(self, variable_scope):
        """
        Counts the number of trainable variables.
        """

        tot_nb_params = 0
        for trainable_variable in variable_scope:
            shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
            #print(shape)
            current_nb_params = self.get_nb_params_shape(shape)
            tot_nb_params = tot_nb_params + current_nb_params
        return tot_nb_params

    def get_nb_params_shape(self, shape):
        '''
        Computes the total number of params for a given shap.
        Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
        '''
        nb_params = 1
        for dim in shape:
            nb_params = nb_params*int(dim)
        return nb_params 
