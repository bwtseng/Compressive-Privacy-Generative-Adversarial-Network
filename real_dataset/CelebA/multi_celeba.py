import math 
import os
import h5py 
import random
import time
import numpy as np 
import pandas as pd 
from numpy.linalg import pinv 
import matplotlib.pyplot as plt
from functools import reduce
from scipy.misc import imread, imresize ,imsave
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils import check_random_state
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
import tensorflow as tf
import tensorflow.contrib.layers as ly 
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes

plt.switch_backend('agg')
tf.set_random_seed(9)

class CPGAN: 
    def __init__(self,args):

        self.arg = args
        self.com_dim = self.arg.com_dim
        self.batch_size = self.arg.batch_size
        self.gamma = self.arg.gamma
        self.mapping_dim = self.arg.mapping_dim
        self.seed = 9

        self.t_data , self.v_data, self.te_data, self.t_label, self.v_label, self.te_label = self.load_data()

        print("Training data size: {} and {}".format(len(self.t_data), len(self.t_label[0])))
        print("Validation data size: {} and {}".format(len(self.v_data), len(self.v_label[0])))
        print("Testing data size: {} and {}".format(len(self.te_data), len(self.te_label[0])))
      
        print(len(self.t_label))
        print(len(self.t_label[0]))
        #assert len(self.t_label[0]) == len(self.t_label[1])

        self.build_model()

        print("The number of multiplication and addtion : {} and {}.".format(self.g_multiplication,self.g_addition))
        print("The number of multiplication and addtion : {} and {}.".format(self.c_multiplication,self.c_addition))


        #qq = self.count_number_trainable_params()
        classes = [9, 10, 5, 2, 4, 5, 3, 2] 
        utility_loss_list = []
        for i in range(40):
            utility_loss_list.append(tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot_list[i],
                                                                                         logits=self.logit_list[i]))
        
        utility_loss = tf.reduce_mean(tf.add_n(utility_loss_list))
        self.loss_c = utility_loss
        self.loss_r_nn = tf.losses.mean_squared_error(self.image_p, self.recon_nn) 
        self.loss_r_lrr = tf.losses.mean_squared_error(self.image_p, self.recon_lrr) 
        self.loss_r_krr = tf.losses.mean_squared_error(self.image_p, self.recon_krr) 
        self.loss_g_nn = 1*utility_loss - self.loss_r_nn

        # base = 10000 (fail) 
        # middle = 50000 (fail)
        # other = 80000(success)
        # upper = 100000(success)
        ### 10 has tested (lambda) !! 

        self.loss_g_lrr = 10*utility_loss - self.loss_r_lrr
        self.loss_g_krr = 10*utility_loss - self.loss_r_krr
        self.theta_r_nn = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='adversary_nn')
        self.theta_r_lrr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='adversary_lrr')
        self.theta_r_krr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='adversary_krr')

        self.theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='privatizer')
        self.theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='utility_classifier')

        # ******************************
        # Bulid assign operation  tensor 
        # ******************************
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

        ###****************************************************

        print('The numbers of parameters in variable_scope G are : {}'.format(self.count_number_trainable_params(self.theta_g)))
        print('The numbers of parameters in variable_scope R are : {}'.format(self.count_number_trainable_params(self.theta_c)))

        uti_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(uti_update):

            self.c_op = tf.train.AdamOptimizer(0.001)
            self.c_opt = self.c_op.minimize(self.loss_c, var_list=self.theta_c)

            self.r_op = tf.train.AdamOptimizer(0.001)
            self.r_opt = self.r_op.minimize(self.loss_r_nn, var_list=self.theta_r_nn)

            self.g_op_nn = tf.train.AdamOptimizer(0.001)
            self.g_opt_nn = self.g_op_nn.minimize(self.loss_g_nn, var_list=self.theta_g)

            self.g_op_lrr = tf.train.AdamOptimizer(0.001)
            self.g_opt_lrr = self.g_op_lrr.minimize(self.loss_g_lrr, var_list=self.theta_g)

            self.g_op_krr = tf.train.AdamOptimizer(0.001)
            self.g_opt_krr = self.g_op_krr.minimize(self.loss_g_krr, var_list=self.theta_g)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=5)

    def load_data(self):

        label_list = []
        subnet_1 = ['Black_Hair', 'Blond_Hair', 'Blurry', 'Eyeglasses', 'Gray_Hair', 'Pale_Skin','Straight_Hair','Wearing_Hat']
        label_list.append(subnet_1)
        subnet_2 = ['Attractive','Bangs','Brown_Hair','Heavy_Makeup','High_Cheekbones','Mouth_Slightly_Open','No_Beard','Oval_Face',\
                    'Pointy_Nose','Rosy_Cheeks', 'Smiling', 'Wavy_Hair', 'Wearing_Lipstick', 'Young']
        label_list.append(subnet_2)
        subnet_3 = ['5_o_Clock_Shadow','Arched_Eyebrows','Bags_Under_Eyes','Bald','Big_Lips','Big_Nose','Bushy_Eyebrows',\
                    'Chubby','Double_Chin','Goatee','Male','Mustache','Narrow_Eyes','Receding_Hairline','Sideburns',\
                    'Wearing_Earrings','Wearing_Necklace','Wearing_Necktie']
        label_list.append(subnet_3)
        img_path_list =list(np.sort(os.listdir(self.arg.data_dir)))
        img_indices = [i for i in range(len(temp))]
        img_list = []
        count = 0
        for i in img_path_list : 
            img_list.append(os.path.join(self.arg.data_dir, i))

        train_img, test_img, original_indices, test_indices = train_test_split(img_list, img_indices, test_size=0.1, 
                                                                        random_state=9, shuffle=False)
        train_img, val_img, train_indices, val_indices = train_test_split(train_img, original_indices, test_size=0.1, 
                                                                        random_state=9, shuffle=False)
        t_lab = []
        lab_40_test = [] 
        v_lab = []
        start = time.time()
        for subnet in label_list: 
            for attr in subnet :
                label_train = []
                label_val = []
                temp = pd.read_csv(self.arg.label_dir)[[attr]].values.reshape(-1)
                temp = np.array(temp)
                train_set = temp[original_indices]
                test_set = temp[test_indices]

                train_set[train_set==-1] = 0
                test_set[test_set==-1] = 0

                val_set = train_set[val_indices]
                train_set = train_set[train_indices]

                t_lab.append(list(train_set))
                v_lab.append(list(val_set))
                lab_40_test.append(list(test_set))

        end = time.time()
        print('Load successfully !!! It costs about {:.3f} sec'.format(end-start))
        return train_img, val_img, test_img, t_lab, v_lab, lab_40_test

    def preprocess(self,data):
        a = []
        for i in  data: 
            temp = self.plot(i)
            temp = imresize(i,(175,175))
            temp = (temp/127.5)-1
            a.append(temp)
        return a


    def bo_batch_norm(self,x, is_training, momentum=0.9, epsilon=0.0001):
        x = tf.layers.batch_normalization(x, momentum=momentum, epsilon=epsilon, training=is_training)
        x = tf.nn.relu(x)
        return x

    def generator_conv(self, image, reuse=False):
        with tf.variable_scope('privatizer') as scope:
            if reuse:
                scope.reuse_variables()  

            self.g_addition = 0
            self.g_multiplication = 0

            channel = image.get_shape().as_list()[-1]

            conv1 = ly.conv2d(image, 64, kernel_size=3, padding='VALID', stride=2, activation_fn=tf.nn.relu, 
                            weights_initializer=tf.contrib.layers.xavier_initializer())

            temp = conv1.get_shape().as_list()
            self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
            self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel)

            conv1 = ly.max_pool2d(conv1, kernel_size=3, padding='VALID', stride=2)

            #conv2 = ly.conv2d(conv1, 256, kernel_size=5, padding='SAME', stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())

            channel = conv1 .get_shape().as_list()[-1]

            conv2 = ly.conv2d(conv1, 64, kernel_size=5, padding='SAME', stride=1, activation_fn=tf.nn.relu, 
                            weights_initializer=tf.contrib.layers.xavier_initializer())

            temp = conv1.get_shape().as_list()
            self.g_addition += (temp[1]*temp[2]*temp[3]*5*5*channel)
            self.g_multiplication += (temp[1]*temp[2]*temp[3]*5*5*channel)

            conv2 = ly.max_pool2d(conv2, kernel_size=3, padding='SAME', stride=2)

            with tf.variable_scope('subenet_first_group'):
                with tf.variable_scope('branch_1'):
                    pool_1 = ly.max_pool2d(conv2, kernel_size=3, padding='SAME', stride=2)

                    channel = pool_1.get_shape().as_list()[-1]

                    conv_1 = ly.conv2d(pool_1, 64, kernel_size=3, padding='SAME',stride=2, 
                                        weights_initializer=tf.contrib.layers.xavier_initializer())

                    temp = conv_1.get_shape().as_list()
                    self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
                    self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel)

                with tf.variable_scope('branch_2'):

                    channel = conv2.get_shape().as_list()[-1]

                    reduce_3 = ly.conv2d(conv2, 64, kernel_size=1, padding='SAME', stride=2,
                                        weights_initializer=tf.contrib.layers.xavier_initializer())

                    temp = reduce_3.get_shape().as_list()
                    self.g_addition += (temp[1]*temp[2]*temp[3]*1*1*channel)
                    self.g_multiplication += (temp[1]*temp[2]*temp[3]*1*1*channel)

                    channel = reduce_3.get_shape().as_list()[-1]
                    conv_3 = ly.conv2d(reduce_3, 64, kernel_size=3, padding='SAME', stride=2,
                                        weights_initializer=tf.contrib.layers.xavier_initializer())

                    temp = conv_3.get_shape().as_list()
                    self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
                    self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel)

                with tf.variable_scope('branch_3'):

                    channel = conv2.get_shape().as_list()[-1]

                    reduce_5 = ly.conv2d(conv2, 64, kernel_size=1, padding='SAME', stride=2,
                                         weights_initializer=tf.contrib.layers.xavier_initializer())

                    temp = reduce_5.get_shape().as_list()
                    self.g_addition += (temp[1]*temp[2]*temp[3]*1*1*channel)
                    self.g_multiplication += (temp[1]*temp[2]*temp[3]*1*1*channel)

                    channel = reduce_5.get_shape().as_list()[-1]

                    conv_5 = ly.conv2d(reduce_5, 64, kernel_size=3, padding='SAME', stride=2,
                                         weights_initializer=tf.contrib.layers.xavier_initializer())

                    temp = conv_5.get_shape().as_list()
                    self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
                    self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel)

                output = tf.concat([conv_1,conv_3,conv_5],axis=2)
                pool3_a = ly.max_pool2d(output,kernel_size=3, padding='SAME', stride=2)
                flat_first = ly.flatten(pool3_a)

                x_shape = flat_first.get_shape().as_list()
                flat_first = ly.fully_connected(flat_first, self.com_dim, activation_fn=tf.nn.relu, 
                                            weights_initializer=tf.contrib.layers.xavier_initializer())

                self.g_addition += (x_shape[-1]*64)
                self.g_multiplication += (x_shape[-1]*64)

            with tf.variable_scope('subnet_second_group'):
                with tf.variable_scope('branch_1'):
                    pool_1 = ly.max_pool2d(conv2, kernel_size=3, padding='SAME', stride=2)

                    channel = pool_1.get_shape().as_list()[-1]

                    conv_1 = ly.conv2d(pool_1, 64, kernel_size=3, padding='SAME',stride=2, 
                                        weights_initializer=tf.contrib.layers.xavier_initializer())

                    temp = conv_1.get_shape().as_list()
                    self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
                    self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel)


                with tf.variable_scope('branch_2'):

                    channel = conv2.get_shape().as_list()[-1]

                    reduce_3 = ly.conv2d(conv2, 64, kernel_size=1, padding='SAME', stride=2,
                                        weights_initializer=tf.contrib.layers.xavier_initializer())

                    temp = reduce_3.get_shape().as_list()
                    self.g_addition += (temp[1]*temp[2]*temp[3]*1*1*channel)
                    self.g_multiplication += (temp[1]*temp[2]*temp[3]*1*1*channel)

                    channel = reduce_3.get_shape().as_list()[-1]
                    conv_3 = ly.conv2d(reduce_3, 64, kernel_size=3, padding='SAME', stride=2,
                                        weights_initializer=tf.contrib.layers.xavier_initializer())

                    temp = conv_3.get_shape().as_list()
                    self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
                    self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel)


                with tf.variable_scope('branch_3'):

                    channel = conv2.get_shape().as_list()[-1]

                    reduce_5 = ly.conv2d(conv2, 64, kernel_size=1, padding='SAME', stride=2,
                                         weights_initializer=tf.contrib.layers.xavier_initializer())

                    temp = reduce_5.get_shape().as_list()
                    self.g_addition += (temp[1]*temp[2]*temp[3]*1*1*channel)
                    self.g_multiplication += (temp[1]*temp[2]*temp[3]*1*1*channel)

                    channel = reduce_5.get_shape().as_list()[-1]

                    conv_5 = ly.conv2d(reduce_5, 64, kernel_size=3, padding='SAME', stride=2, 
                                        weights_initializer=tf.contrib.layers.xavier_initializer())

                    temp = conv_5.get_shape().as_list()
                    self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
                    self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel)

                output = tf.concat([conv_1,conv_3,conv_5],axis=2)
                
                pool3_a = ly.max_pool2d(output, kernel_size=3, padding='SAME', stride=2)
                flat_second = ly.flatten(pool3_a)

                x_shape = flat_second.get_shape().as_list()

                flat_second = ly.fully_connected(flat_second, self.com_dim, activation_fn=tf.nn.relu, 
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

                self.g_addition += (x_shape[-1]*64)
                self.g_multiplication += (x_shape[-1]*64)

            return flat_first, flat_second

    def utility_classifier(self, flat_1, flat_2, reuse=False):

        temp_5 = []
        subnet_1 = ['Black_Hair', 'Blond_Hair', 'Blurry', 'Eyeglasses', 'Gray_Hair', 'Pale_Skin','Straight_Hair','Wearing_Hat']
        temp_5.append(subnet_1)
        subnet_2 = ['Attractive','Bangs','Brown_Hair','Heavy_Makeup','High_Cheekbones','Mouth_Slightly_Open','No_Beard','Oval_Face',\
                    'Pointy_Nose','Rosy_Cheeks', 'Smiling', 'Wavy_Hair', 'Wearing_Lipstick', 'Young']
        temp_5.append(subnet_2)
        subnet_3 = ['5_o_Clock_Shadow','Arched_Eyebrows','Bags_Under_Eyes','Bald','Big_Lips','Big_Nose','Bushy_Eyebrows',\
                    'Chubby','Double_Chin','Goatee','Male','Mustache','Narrow_Eyes','Receding_Hairline','Sideburns',\
                    'Wearing_Earrings','Wearing_Necklace','Wearing_Necktie']
        temp_5.append(subnet_3)

        self.c_addition = 0
        self.c_multiplication = 0

        logit = []
        prob_list = []

        with tf.variable_scope('utility_classifier'):

            with tf.variable_scope('group_1'):

                x_shape = flat_1.get_shape().as_list()

                fc1 = ly.fully_connected(flat_1, 64, activation_fn=tf.nn.relu, 
                                        weights_initializer=tf.contrib.layers.xavier_initializer())

                self.c_addition += (x_shape[-1]*64)
                self.c_multiplication += (x_shape[-1]*64)

                fc1 = tf.nn.dropout(fc1, keep_prob=self.keep_prob)

                x_shape = fc1.get_shape().as_list()

                fc2 = ly.fully_connected(fc1, 64, activation_fn=tf.nn.relu, 
                                        weights_initializer=tf.contrib.layers.xavier_initializer())

                self.c_addition += (x_shape[-1]*64)
                self.c_multiplication += (x_shape[-1]*64)

                fc2 = tf.nn.dropout(fc2, keep_prob=self.keep_prob)
                for i in temp_5[0]:
                    with tf.variable_scope(i):

                        x_shape = fc2.get_shape().as_list()

                        output = ly.fully_connected(fc2, 2, activation_fn = None, 
                                                    weights_initializer=tf.contrib.layers.xavier_initializer())

                        self.c_addition += (x_shape[-1]*2)
                        self.c_multiplication += (x_shape[-1]*2)

                        prob = tf.nn.softmax(output)
                    logit.append(output)
                    prob_list.append(prob)

            with tf.variable_scope('group_2'):

                x_shape = flat_2.get_shape().as_list()

                fc1 = ly.fully_connected(flat_2, 64, activation_fn=tf.nn.relu, 
                                        weights_initializer=tf.contrib.layers.xavier_initializer())

                self.c_addition += (x_shape[-1]*64)
                self.c_multiplication += (x_shape[-1]*64)

                fc1 = tf.nn.dropout(fc1, keep_prob=self.keep_prob)

                x_shape = fc1.get_shape().as_list()

                fc2 = ly.fully_connected(fc1, 64, activation_fn=tf.nn.relu, 
                                        weights_initializer=tf.contrib.layers.xavier_initializer())

                self.c_addition += (x_shape[-1]*64)
                self.c_multiplication += (x_shape[-1]*64)

                fc2 = tf.nn.dropout(fc2, keep_prob=self.keep_prob)
                for i in temp_5[1]:
                    with tf.variable_scope(i):

                        x_shape = fc2.get_shape().as_list()

                        output = ly.fully_connected(fc1, 2, activation_fn = None, 
                                                    weights_initializer=tf.contrib.layers.xavier_initializer())

                        self.c_addition += (x_shape[-1]*2)
                        self.c_multiplication += (x_shape[-1]*2)

                        prob = tf.nn.softmax(output)
                    logit.append(output)
                    prob_list.append(prob)

            with tf.variable_scope('group_3'):      

                x_shape = flat_2.get_shape().as_list()

                fc1 = ly.fully_connected(flat_2, 64, activation_fn=tf.nn.relu, 
                                        weights_initializer=tf.contrib.layers.xavier_initializer())

                self.c_addition += (x_shape[-1]*64)
                self.c_multiplication += (x_shape[-1]*64)

                fc1 = tf.nn.dropout(fc1, keep_prob=self.keep_prob)

                x_shape = fc1.get_shape().as_list()

                fc2 = ly.fully_connected(fc1, 64, activation_fn=tf.nn.relu, 
                                        weights_initializer=tf.contrib.layers.xavier_initializer())

                self.c_addition += (x_shape[-1]*64)
                self.c_multiplication += (x_shape[-1]*64)

                fc2 = tf.nn.dropout(fc2, keep_prob=self.keep_prob)
                for i in temp_5[2]:
                    with tf.variable_scope(i):

                        x_shape = fc2.get_shape().as_list()

                        output = ly.fully_connected(fc1, 2, activation_fn = None, 
                                                    weights_initializer=tf.contrib.layers.xavier_initializer())

                        self.c_addition += (x_shape[-1]*64)
                        self.c_multiplication += (x_shape[-1]*64)

                        prob = tf.nn.softmax(output)
                    logit.append(output)
                    prob_list.append(prob)

        return logit, prob_list

    def adversary_nn(self, final_latent, reuse=False):
        with tf.variable_scope('adversary_nn') as scope:
            if reuse:
                scope.reuse_variables()
            latent = ly.fully_connected(final_latent, 5*5*128, activation_fn=None, 
                                        weights_initializer=tf.contrib.layers.xavier_initializer())
            latent = self.bo_batch_norm(latent, self.is_train)
            latent = tf.reshape(latent, shape=[-1, 5, 5, 128])       
            dim = 32
            latent = ly.conv2d_transpose(latent, dim*4, kernel_size=3, stride=2, padding='SAME', 
                                        activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
            latent = tf.nn.relu(latent)
            upsample1 = ly.conv2d_transpose(latent, dim*4, kernel_size=3, stride=2, padding='VALID', 
                                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
            upsample1 = tf.nn.relu(upsample1)
            upsample2 = ly.conv2d_transpose(upsample1, dim*2, kernel_size=3, stride=2, padding='VALID', 
                                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
            upsample2 = tf.nn.relu(upsample2)
            upsample3 = ly.conv2d_transpose(upsample2, dim*1, kernel_size=3, stride=2, padding='VALID', 
                                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
            upsample3 = tf.nn.relu(upsample3)
            upsample4 = ly.conv2d_transpose(upsample3 ,3, kernel_size=3, stride=2, padding='VALID', 
                                            activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer())
        return upsample4

    # *****************************************
    # If it is center-adjusted, we can directly ignore the biase variable.
    # *****************************************
    def adversary_lrr(self, final_latent, reuse=False):
        with tf.variable_scope('adversary_lrr') as scope:  
            if reuse: 
                scope.reuse_variables()
            recontruction = ly.fully_connected(final_latent, 175*175*3, activation_fn=None, 
                                                weights_initializer=tf.contrib.layers.xavier_initializer(), 
                                                biases_initializer = None)
        return tf.reshape(recontruction, shape=[-1, 175, 175, 3])


    def adversary_krr(self, kernel_map, reuse=False):
        with tf.variable_scope('adversary_krr') as scope:  
            if reuse: 
                scope.reuse_variables()
            recontruction = ly.fully_connected(kernel_map, 175*175*3, activation_fn=None, 
                                            weights_initializer=tf.contrib.layers.xavier_initializer(), 
                                            biases_initializer = None)
        return tf.reshape(recontruction, shape=[-1, 175, 175, 3])

    def _conv(self, input, filter_shape, stride):
        """Convolutional layer for residual block."""
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
        x = tf.nn.relu(x)
        x = self._conv(x, [3, 3, in_filters, out_filters], stride)
        # second convolution layer
        x = self.bo_batch_norm(x,self.is_train)
        x = tf.nn.relu(x)
        x = self._conv(x, [3, 3, out_filters, out_filters], stride)

        if in_filters != out_filters:

            if option == 0:
                difference = out_filters - in_filters
                left_pad = int(difference / 2)
                right_pad = int(difference - left_pad)
                identity = tf.pad(input_, [[0, 0], [0, 0], [0, 0], [left_pad, right_pad]])
                return x + identity
            else:
                print("Not implemented error")
                exit(1)
        else:

            return x + input_

    def init_tensor(self, shape):
        return tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=1.0))

    def residual_g(self,image,reuse=False):
        stride = [1,1,1]
        filter_size = [3,3,3]
        with tf.variable_scope('compressor') as scope:
            if reuse : 
                scope.reuse_variables()  
            x = self._conv(image, [3, 3, 3, 3], 1)      
            for i in range(len(filter_size)):
                for j in range(len([3,3,3])):
                    #with tf.variable_scope('unit_%d_sublayer_%d' % (i, j)):
                        if j == 0:
                            if i == 0:
                                # transition from init stage to the first stage stage
                                #x = self._residual_unit(x, 16, filter_size[i], stride[i])
                                x = self._residual_unit(x, 3, filter_size[i], stride[i])
                            else:
                                x = self._residual_unit(x, filter_size[i - 1], filter_size[i], stride[i])
                        else:
                            x = self._residual_unit(x, filter_size[i], filter_size[i], stride[i])

            return x 


    def RFF_map(self, input_tensor_1, input_tensor_2, seed, stddev, output_dim):
        input_tensor = tf.concat([input_tensor_1, input_tensor_2], axis=1)
        print("Information that the adversary can get: {}".format(input_tensor))
        random_state = check_random_state(seed)
        gamma = stddev
        omega_matrix_shape = [self.arg.dim*2, output_dim]
        bias_shape = [output_dim]

        '''
        *******************************
        This is the source from scikit-learn function RFF_MAP
        ******************************* 
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
        '''
        # *****************************
        # Instead, we use the tensorflow source RFF mapping, please find more detail in the official document.
        # *****************************
        omega_matrix = constant_op.constant(np.sqrt(2 * gamma) *
           random_state.normal(size=omega_matrix_shape),dtype=dtypes.float32)

        bias = constant_op.constant(
            random_state.uniform(
            0.0, 2 * np.pi, size=bias_shape), dtype=dtypes.float32)

        x_omega_plus_bias = math_ops.add(
            math_ops.matmul(input_tensor, omega_matrix), bias)

        return math.sqrt(2.0 / output_dim) * math_ops.cos(x_omega_plus_bias)


    def build_model(self):

        tf.reset_default_graph()

        self.image_p = tf.placeholder(tf.float32,shape=(None, 175, 175, 3))
        self.keep_prob = tf.placeholder(tf.float32)
        self.label_list = [] 
        self.one_hot_list = []
        for i in range(40):
            with tf.name_scope('label_placeholder_'+str(i+1)):
                label_p = tf.placeholder(tf.int64,shape=[None])
                #label_p = tf.placeholder(tf.float32, shape=[None,1])
                one_hot = tf.one_hot(label_p,2)
                self.label_list.append(label_p)
                self.one_hot_list.append(one_hot)

        self.noise_p = tf.placeholder(tf.float32, shape=(None,175, 175, 3))
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_train = tf.placeholder(tf.bool)

        # Privatizer
        if self.arg.noise: 
            self.noisy_img = tf.add(self.image_p,self.noise_p)
            self.latent, self.latent_1 = self.generator_conv(self.noisy_img)
        else : 
            self.latent ,self.latent_1 = self.generator_conv(self.image_p)

        ## Classifier
        self.logit_list , self.prob_list = self.utility_classifier(self.latent, self.latent_1)

        # Weight placeholder
        self.lrr_mu_p = tf.placeholder(tf.float32, shape=[self.com_dim*2])
        self.krr_mu_p = tf.placeholder(tf.float32, shape=[self.mapping_dim])
        self.t_mu_p = tf.placeholder(tf.float32, shape=[175*175*3])
        self.krr_weights = tf.placeholder(tf.float32, shape=[self.mapping_dim, 175*175*3])
        self.lrr_weights = tf.placeholder(tf.float32, shape=[self.com_dim*2, 175*175*3])
        self.lrr_mu = self.init_tensor([self.com_dim*2])
        self.krr_mu = self.init_tensor([self.mapping_dim])  
        self.t_mu = self.init_tensor([175*175*3])

        # If applyint normalization mechansim, one can directly remove the comment symbol.
        # This is weill remain the optimal solution, which is refered from Kung's Kernel Method and Machine Learning.
        self.kernel_map = self.RFF_map(self.latent, self.latent_1, self.seed, self.gamma, self.mapping_dim)
        #self.kernel_map_deduct = self.kernel_map - self.krr_mu
        self.latent_concat = tf.concat([self.latent, self.latent_1], axis=1)
        self.recon_nn = self.adversary_nn(self.latent_concat)
        #self.recon_nn = self.adversary_nn(self.latent, self.latent_1)
        self.recon_lrr = self.adversary_lrr(self.latent_concat)
        #self.recon_lrr = self.adversary_lrr(self.latent, self.latent_1, self.lrr_mu)
        #self.recon_krr = self.adversary_krr(self.kernel_map_deduct)
        self.recon_krr = self.adversary_krr(self.kernel_map)
        #self.recon_lrr = self.recon_lrr + tf.reshape(self.t_mu, [175, 175, 3])
        #self.recon_krr = self.recon_krr + tf.reshape(self.t_mu, [175, 175, 3])

    def plot(self,x):
        x = x - np.min(x)
        x = x /  np.max(x)
        x *= 255  
        x = x.astype(np.uint8)
        x = x.reshape(175, 175, 3)
        return x 


    def plot_10slot(self, name="Reconsturcted_images.png"):
        r ,c  = 2,10
        random_sample_img = np.array(self.te_data[:128])
        random_sample_label = np.array([i for i in range(128)])
        penal = np.array([[0.5,1] for i in range(128)])
        no = np.random.normal(size=(128, 112, 112, 3))
        compress_representations = self.sess.run(self.latent_concat, feed_dict={
                                                                        self.image_p:random_sample_img.reshape(128, 175, 175,3), 
                                                                        self.label_p:random_sample_label, 
                                                                        self.noise_p:no, 
                                                                        self.keep_prob:1, 
                                                                        self.penalty:penal})
        reconstructions = self.sess.run(self.up, feed_dict={self.latent_no:compress_representations})

        plt.figure(figsize=(10, 2))
        n = 10

        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(self.plot(self.te_data[i]))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(self.plot(reconstructions[i]))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.savefig(os.path.join(self.arg.model_dir, name))
        plt.close()
        #plt.show()

    def resize(self, image):
        img_list = []
        for img in image : 
            img_list.append(self.preprocess(imresize(img, (175, 175))))
        return a 

    def read(self, image):
        img_list = []
        for img in image : 
            img_list.append(imread(img))
        return img_list

    def plot_175(self,x):
        x = x - np.min(x)
        x = x /  np.max(x)
        x *= 255  
        x= x.astype(np.uint8)
        x = x.reshape(175, 175, 3)
        return x 


    def compute_adv_mse(self, data, label):
        
        # *************************************
        # multiple adversaries strategy: choose the adversary with minimum reconstruction loss,
        # so that we are able to robuslty train our CPGAN.
        # *************************************

        mse_nn = []
        mse_lrr = []
        mse_krr = []
        for batch_x, batch_y in self.next_batch(data, label, shuffle=False, batch_size=self.batch_size):
            b = batch_x.shape[0]
            no = np.random.normal(size=(b, 175, 175, 3))
            batch_x = batch_x.reshape(b ,175, 175, 3)
            feed_dict = {}
            feed_dict[self.image_p] = batch_x
            feed_dict[self.noise_p] = no
            feed_dict[self.is_train] = False
            reconstruction_nn = self.sess.run(self.recon_nn, feed_dict=feed_dict)
            reconstruction_lrr = self.sess.run(self.recon_lrr, feed_dict=feed_dict)
            reconstruction_krr = self.sess.run(self.recon_krr, feed_dict=feed_dict)

            for k in range(len(up_nn)):
                error_nn.append(mean_squared_error(batch_x[k].flatten(), reconstruction_nn[k].flatten()))
                error_lrr.append(mean_squared_error(batch_x[k].flatten(), reconstruction_lrr[k].flatten()))
                error_krr.append(mean_squared_error(batch_x[k].flatten(), reconstruction_krr[k].flatten()))

        # Save adversary reconsturciton.
        imsave('original.png', self.plot_175(i[0]))
        imsave('nn_reco.png', self.plot_175(up_nn[0]))
        imsave('lrr_reco.png', self.plot_175(up_lrr[0]))
        imsave('krr_reco.png', self.plot_175(up_krr[0]))

        return np.mean(mse_nn), np.mean(mse_lrr), np.mean(mse_krr)

    def kernel_rbf(self, x, y): 
        b = len(self.t_data)
        K = euclidean_distances(x, y, squared=True)
        gamma = 1.0 / b
        K *= -(2*gamma)
        return np.exp(K)

    def kernel_matrix(self):
        # Build kernel matrix from training dataset!
        emb_list = []
        for batch_x, batch_y in self.next_batch(self.t_data, self.t_label, shuffle = False, batch_size = self.batch_size):
            b = batch_x.shape[0]
            feed_dict = {}
            feed_dict[self.image_p] = batch_x
            feed_dict[self.noise_p] = noise
            feed_dict[self.is_train] = False
            feed_dict[self.keep_prob] = 1
            noise = np.random.normal(size=(128,175,175,3))
            compressing_representation = self.sess.run(self.latent, feed_dict=feed_dict)
            emb_list.append(uu)

        temp = 0
        for i in emb_list: 
            if temp == 0: 
                emb_matrix = i 
                temp +=1 
                continue
            emb_matrix = np.concatenate((emb_matrix,i),axis=0)

        K = np.zeros((len(self.t_data), len(self.t_data)))

        for i in range(len(self.t_data)):
            for j in range(i):
                #D[i,j]=quadraticChiDist(X[i,:],X[j,:])
                K[i,j] = self.kernel_rbf(emb_matrix[i,:], emb_matrix[j:])
                K[j,i] = K[i,j]
                # You can also implement it from instrinsic space, and use their inner product to generate kernel matrix 
                # Reference: Kernel Method and Machine Learning
        return K


    def operation_degree(self, m):
        init = 1 
        for i in range(1,m+1) : 
            init *= i
        return init 

    def KRR_close_form(self, emb_matrix, train_matrix, train_mu):
        # Use the random fourier transform to approximate the RBF kernel 
        # Note that the training data is too large so that we use the intrinsic space mapping 
        # And use the tensorflow conrtrib package to get the RFF mapping rather than hand crafting
        # More information refers to https://github.com/hichamjanati/srf  

        rau = 0.001
        mu = np.mean(emb_matrix, axis=0)
        #emb_matrix = emb_matrix - mu
        emb_matrix = emb_matrix.T 
        s = np.dot(emb_matrix, emb_matrix.T)
        a,b = s.shape
        identity = np.identity(a)
        s_inv = np.linalg.inv(s + rau * np.identity(a))
        #train_norm = train_matrix - train_mu
        weights = np.dot(np.dot(s_inv, emb_matrix), train_matrix)
        print('Shape of the KRR weight: {}.'.format(weights.shape))
        return weights, mu 


    def LRR_close_form(self, emb_matrix, train_matrix, train_mu):#, compute=True):
        mu = np.mean(emb_matrix, axis=0)
        #emb_matrix = emb_matrix - mu
        emb_matrix = emb_matrix.T
        rau = 0.001
        s = np.dot(emb_matrix, emb_matrix.T)
        h,w = s.shape
        s_inv = np.linalg.inv(s + rau*np.identity(h))
        #train_norm = train_matrix - train_mu
        weights = np.dot(np.dot(s_inv, emb_matrix), train_matrix)
        print('Shape of the LRR weights: {}'.format(weights.shape))
        return weights, mu

    def get_emb_matrix(self): 
        count = 0
        for batch_x, batch_y in self.next_batch(self.t_data, self.t_label, self.batch_size):
            b = batch_x.shape[0]
            no = np.random.normal(size=(128, 175, 175, 3))
            batch_x = batch_x.reshape(b, 175, 175, 3)
            feed_dict = {}
            feed_dict[self.image_p] = batch_x
            feed_dict[self.keep_prob] = 1
            feed_dict[self.is_train] = False
            compressing_representation_concat, kernel_map = self.sess.run([self.latent_concat, self.kernel_map], feed_dict=feed_dict)
            if count == 0 : 
                emb_matrix_lrr = compressing_representation_concat
                emb_matrix_krr = kernel_map
                count+=1 
            else : 
                emb_matrix_lrr = np.concatenate((emb_matrix_lrr, compressing_representation_concat), axis=0)
                emb_matrix_krr = np.concatenate((emb_matrix_krr, kernel_map), axis=0)
                count+=1 
        print('Successfully get embedding matrix.')   
        return emb_matrix_lrr, emb_matrix_krr


    def get_train_matrix(self): 
        real_list = []
        count = 0 
        temp = [] 
        for i in self.t_data : 
            k = imresize(self.plot(imread(i)), (175,175))
            k = (k/127.5) -1
            temp.append(k.flatten().reshape(1, 175*175*3))

        train_matrix = np.concatenate(temp, axis=0)
        print('Successfully get flatted train matrix !!!!')     
        return train_matrix

    def assign(self, train_matrix, train_mu):
        feed_dict_assign = {}
        emb_matrix_lrr, emb_matrix_krr = self.get_emb_matrix()

        error_list = []
        update_choice = [self.g_opt_nn, self.g_opt_lrr, self.g_opt_krr]

        lrr_weights, lrr_mu = self.LRR_close_form(emb_matrix_lrr, train_matrix, train_mu)
        feed_dict_assign[self.lrr_mu_p] = lrr_mu
        feed_dict_assign[self.lrr_weights] = lrr_weights

        krr_weights, krr_mu = self.KRR_close_form(emb_matrix_krr, train_matrix, train_mu)
        feed_dict_assign[self.krr_mu_p] = krr_mu
        feed_dict_assign[self.krr_weights] = krr_weights
        feed_dict_assign[self.t_mu_p] = train_mu

        self.sess.run(self.assign_op, feed_dict = feed_dict_assign)

        error_nn, error_lrr, error_krr = self.compute_adv_mse(self.v_data, self.v_label)
        error_list.append(error_nn) 
        error_list.append(error_lrr)
        error_list.append(error_krr)
        print('Average MSE among all testing images is {:.3f}, {:.3f}, {:.3f}.(nn,lrr,krr)'.format(error_nn, 
                                                                                                   error_lrr, error_krr))
        optimize_g = update_choice[np.argmin(error_list)]

        return optimize_g, feed_dict_assign


    def train(self):

        acc_trace = [] 
        mse_trace = [] 
        mse_trace_1 = []
        # **********************************************************************************
        # For writting log file, but python provide a more convienent package msglogerr (?)
        # os.mkdir('Male_2_test')
        # f = open('Male_2_test/Male_noise_log.txt','w')
        # **********************************************************************************
        loss_trace = []
        train_matrix = self.get_train_matrix()
        train_mu = np.mean(train_matrix, axis=0)
        is_best = 0
        # update_option = [self.g_opt_nn, self.g_opt_lrr, self.g_opt_krr]
        for epo in range(self.arg.epoch):
            epoch_loss = []
            start_epo = time.time()

            for batch_x, batch_y in self.next_batch(self.t_data, self.t_label, shuffle=True, self.batch_size):
                b = batch_x.shape[0]
                no = np.random.normal(size=(b, 175, 175, 3)) # Laplician noise cound also be taken into consideration.
                feed_dict = {}
                feed_dict[self.image_p] = batch_x

                for attr in range(40):
                    feed_dict[self.label_list[attr]] = np.array(batch_y[attr]).reshape(-1)
                feed_dict[self.keep_prob] = 1
                feed_dict[self.is_train] = True
                feed_dict[self.noise_p] = no

                for _ in range(self.arg.citer):
                    _ = self.sess.run(self.r_opt, feed_dict = feed_dict)
                c_loss , _ = self.sess.run([self.loss_c, self.c_opt], feed_dict = feed_dict)

            end = time.time()

            print("Training for R and C costs about {}.".format(end-start))

            start_g = time.time()

            optimize_g, feed_dict = self.assign(train_matrix, train_mu)
            
            for batch_x, batch_y in self.next_batch(self.t_data, self.t_label, shuffle=False, batch_size=self.batch_size):
                b = batch_x.shape[0]
                no = np.random.normal(size=(b, 175, 175,3)) # Laplician noise cound also be taken into consideration.
                feed_dict[self.image_p] = batch_x
                for attr in range(40):
                    feed_dict[self.label_list[attr]] = np.array(batch_y[attr]).reshape(-1)
                feed_dict[self.keep_prob] = 1
                feed_dict[self.is_train] = True
                feed_dict[self.noise_p] = no
                for _ in range(1):
                    _  = self.sess.run([optimize_g], feed_dict=feed_dict)

            end = time.time()
            print("Training privatizer costs about {:.3f} sec.".format(end - start_g))

            #acc_testing = self.compute_acc(self.te_data, self.te_label)
            acc_validation = self.compute_acc(self.v_data, self.v_label)
            print('Epoch [{}/{}], cost {} sec, validation acc {}'.format(epo+1, self.arg.epoch, end-start_epo, 
                                                                                         acc_validation))

            if acc_validation > is_best:
                is_best = acc_validation
                self.saver.save(self.sess, os.path.join(self.arg.model_dir, self.arg.name+"_ckpt_best"))
                self.save_g()
            if (epo+1) % 3 == 0:
                self.saver.save(self.sess, os.path.join(self.arg.model_dir, self.arg.name+"_ckpt_"+str(epo+1)))
                self.plot_10slot()      

    def save_g(self):

        if self.arg.noise : 
            np.save(os.path.join(self.arg.model_dir, self.arg.name+"_weights_"+str(self.arg.com_dim)+"_noise.npy"), 
                    self.sess.run(self.theta_g))
        else : 
            np.save(os.path.join(self.arg.model_dir, self.arg.name+"_weights_"+str(self.arg.com_dim)+".npy"), 
                    self.sess.run(self.theta_g))



    def shuffle(self):
        ### take all list in different into each personal list
        start = time.time()
        temp = [[] for i in range(len(self.t_data))]
        for i in range(len(temp)):
            temp[i].append(self.t_data[i])
            for k in range(40):
                temp[i].append(self.t_label[k][i])
        
        #a = list(zip(self.t_data,self.t))

        random.shuffle(temp)
        temp_img = []
        temp_1 = [[] for i in range(40)]
        count = 1
        for i in range(len(self.t_data)):
            temp_img.append(temp[i][0])
            for k in range(1,41):
                temp_1[k-1].append(temp[i][k])
        end = time.time()
        print('Shuffling needs about {} sec.'.format(end-start))

        return temp_img, temp_1


    def next_batch(self, data, label, shuffle=False, batch_size=256):
        data_size = len(data)
        iter_num = data_size//batch_size
        data_rest_num = data_size - (epo*batch_size)
        if shuffle:
            data_zip = list(zip(data, label))
            random.shuffle(data_zip)
            data , label = zip(*data_zip)
        for i in range(0, data_size, batch_size):
            if i ==  (epo *batch_size) : 
                label40__list = [[] for i in range(40)]
                for k in range(40):
                    label40_list[k].append(label[k][i:]) 

                yield np.array(self.preprocess(self.read(data[i:]))), oo  

            else : 
                label40_list = [[] for i in range(40)]
                for k in range(40):
                    label40_list[k].append(label[k][i: i+batch_size])  

                yield np.array(self.preprocess(self.read(data[i: i+batch_size]))), label40_list


    def compute_acc(self, data, label):
        acc_list = []
        pred_prob_list = [[] for i in range(40)]
        for batch_x, batch_y in self.next_batch(data, label, shuffle=False, batch_size = self.batch_size):
            b = batch_x.shape[0]
            no = np.random.normal(size=(b, 175, 175, 3))
            feed_dict = {}
            feed_dict[self.image_p] = batch_x.reshape(b, 175, 175, 3)
            feed_dict[self.is_train] = False
            feed_dict[self.keep_prob] = 1
            feed_dict[self.noise_p] = no
            pred_prob_list = self.sess.run(self.prob_list, feed_dict=feed_dict)
            for i in range(40):
                pred_prob_list[i].append(temp[i])

        for i in range(40):
            temp = np.concatenate(pred_prob_list[i], axis=0)
            acc = accuracy_score(np.argmax(temp, 1), label[i])
            acc_list.append(acc)
        return acc_list

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

    def test(self):
        # only consider the best checkpoint, if you want double check other models, please revise the following line.
        self.saver.restore(self.sess, os.path.join(self.arg.model_dir, self.arg.name+"_ckpt_best"))
        acc_testing = self.compute_acc(self.te_data,self.te_label)
        print('Testing accuracy: {:.3f}'.format(acc_testing))