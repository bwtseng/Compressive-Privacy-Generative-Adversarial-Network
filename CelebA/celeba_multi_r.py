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

class cpgan: 
    def __init__(self,args):

        self.arg = args
        self.com_dim = self.arg.com_dim
        self.batch_size = self.arg.batch_size
        self.path = self.arg.path
        self.path_label = self.arg.path_label
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
            utility_loss_list.append(tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot_list[i],\
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

        self.theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='compressor')
        self.theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='utility_classifier')

        ### Assign operation  ********************************
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
        temp =list(np.sort(os.listdir(self.arg.path)))
        indices = [i for i in range(len(temp))]
        img = []
        count = 0
        for i in temp : 
            img.append(os.path.join(self.arg.path, i))

        t_img, te_img, original_indices, test_indices = train_test_split(img, indices, test_size=0.1, random_state=9, shuffle=False)
        t_img, val_img, train_indices, val_indices = train_test_split(t_img, original_indices, test_size=0.1, random_state=9, shuffle=False)
        t_lab = []
        lab_40_test = [] 
        v_lab = []
        start = time.time()
        pp = 0
        for j in temp_5: 
            for q in j :
                #print(pp)
                lab_t = []
                lab_v = []
                temp = pd.read_csv(self.path_label)[[q]].values.reshape(-1)
                temp = np.array(temp)
                ori = temp[original_indices]
                tes = temp[test_indices]

                ori[ori==-1] = 0
                tes[tes==-1] = 0

                val = ori[val_indices]
                tra = ori[train_indices]

                t_lab.append(list(tra))
                v_lab.append(list(val))
                lab_40_test.append(list(tes))
                pp+=1

        end = time.time()
        print('Load successfully!!! And cost about {} sec'.format(end-start))
        return t_img, val_img, te_img, t_lab, v_lab, lab_40_test

    def preprocess(self,data):
        a=[]
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
        with tf.variable_scope('compressor') as scope:
            if reuse:
                scope.reuse_variables()  

            self.g_addition = 0
            self.g_multiplication = 0

            channel = image.get_shape().as_list()[-1]

            conv1 = ly.conv2d(image, 64, kernel_size=3, padding='VALID', stride=2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())

            temp = conv1.get_shape().as_list()
            self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
            self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel)

            conv1 = ly.max_pool2d(conv1, kernel_size=3, padding='VALID', stride=2)

            #conv2 = ly.conv2d(conv1, 256, kernel_size=5, padding='SAME', stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())

            channel = conv1 .get_shape().as_list()[-1]

            conv2 = ly.conv2d(conv1, 64, kernel_size=5, padding='SAME', stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())

            temp = conv1.get_shape().as_list()
            self.g_addition += (temp[1]*temp[2]*temp[3]*5*5*channel)
            self.g_multiplication += (temp[1]*temp[2]*temp[3]*5*5*channel)

            conv2 = ly.max_pool2d(conv2, kernel_size=3, padding='SAME', stride=2)

            with tf.variable_scope('subenet_first_group'):
                with tf.variable_scope('branch_1'):
                    pool_1 = ly.max_pool2d(conv2, kernel_size=3, padding='SAME', stride=2)

                    channel = pool_1.get_shape().as_list()[-1]

                    conv_1 = ly.conv2d(pool_1, 64, kernel_size=3, padding='SAME',stride=2, weights_initializer=tf.contrib.layers.xavier_initializer())

                    temp = conv_1.get_shape().as_list()
                    self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
                    self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel)

                with tf.variable_scope('branch_2'):

                    channel = conv2.get_shape().as_list()[-1]

                    reduce_3 = ly.conv2d(conv2, 64, kernel_size=1, padding='SAME', stride=2,weights_initializer=tf.contrib.layers.xavier_initializer())

                    temp = reduce_3.get_shape().as_list()
                    self.g_addition += (temp[1]*temp[2]*temp[3]*1*1*channel)
                    self.g_multiplication += (temp[1]*temp[2]*temp[3]*1*1*channel)

                    channel = reduce_3.get_shape().as_list()[-1]
                    conv_3 = ly.conv2d(reduce_3, 64, kernel_size=3, padding='SAME', stride=2,weights_initializer=tf.contrib.layers.xavier_initializer())

                    temp = conv_3.get_shape().as_list()
                    self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
                    self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel)

                with tf.variable_scope('branch_3'):

                    channel = conv2.get_shape().as_list()[-1]

                    reduce_5 = ly.conv2d(conv2, 64, kernel_size=1, padding='SAME', stride=2, weights_initializer=tf.contrib.layers.xavier_initializer())

                    temp = reduce_5.get_shape().as_list()
                    self.g_addition += (temp[1]*temp[2]*temp[3]*1*1*channel)
                    self.g_multiplication += (temp[1]*temp[2]*temp[3]*1*1*channel)

                    channel = reduce_5.get_shape().as_list()[-1]

                    conv_5 = ly.conv2d(reduce_5, 64, kernel_size=3, padding='SAME', stride=2, weights_initializer=tf.contrib.layers.xavier_initializer())

                    temp = conv_5.get_shape().as_list()
                    self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
                    self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel)

                output = tf.concat([conv_1,conv_3,conv_5],axis=2)
                pool3_a = ly.max_pool2d(output,kernel_size=3, padding='SAME', stride=2)
                flat_first = ly.flatten(pool3_a)

                x_shape = flat_first.get_shape().as_list()
                flat_first = ly.fully_connected(flat_first, self.com_dim, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())

                self.g_addition += (x_shape[-1]*64)
                self.g_multiplication += (x_shape[-1]*64)

            with tf.variable_scope('subnet_second_group'):
                with tf.variable_scope('branch_1'):
                    pool_1 = ly.max_pool2d(conv2, kernel_size=3, padding='SAME', stride=2)

                    channel = pool_1.get_shape().as_list()[-1]

                    conv_1 = ly.conv2d(pool_1, 64, kernel_size=3, padding='SAME',stride=2, weights_initializer=tf.contrib.layers.xavier_initializer())

                    temp = conv_1.get_shape().as_list()
                    self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
                    self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel)


                with tf.variable_scope('branch_2'):

                    channel = conv2.get_shape().as_list()[-1]

                    reduce_3 = ly.conv2d(conv2, 64, kernel_size=1, padding='SAME', stride=2,weights_initializer=tf.contrib.layers.xavier_initializer())

                    temp = reduce_3.get_shape().as_list()
                    self.g_addition += (temp[1]*temp[2]*temp[3]*1*1*channel)
                    self.g_multiplication += (temp[1]*temp[2]*temp[3]*1*1*channel)

                    channel = reduce_3.get_shape().as_list()[-1]
                    conv_3 = ly.conv2d(reduce_3, 64, kernel_size=3, padding='SAME', stride=2,weights_initializer=tf.contrib.layers.xavier_initializer())

                    temp = conv_3.get_shape().as_list()
                    self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
                    self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel)


                with tf.variable_scope('branch_3'):

                    channel = conv2.get_shape().as_list()[-1]

                    reduce_5 = ly.conv2d(conv2, 64, kernel_size=1, padding='SAME', stride=2, weights_initializer=tf.contrib.layers.xavier_initializer())

                    temp = reduce_5.get_shape().as_list()
                    self.g_addition += (temp[1]*temp[2]*temp[3]*1*1*channel)
                    self.g_multiplication += (temp[1]*temp[2]*temp[3]*1*1*channel)

                    channel = reduce_5.get_shape().as_list()[-1]

                    conv_5 = ly.conv2d(reduce_5, 64, kernel_size=3, padding='SAME', stride=2, weights_initializer=tf.contrib.layers.xavier_initializer())

                    temp = conv_5.get_shape().as_list()
                    self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
                    self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel)

                output = tf.concat([conv_1,conv_3,conv_5],axis=2)
                
                pool3_a = ly.max_pool2d(output, kernel_size=3, padding='SAME', stride=2)
                flat_second = ly.flatten(pool3_a)

                x_shape = flat_second.get_shape().as_list()

                flat_second = ly.fully_connected(flat_second, self.com_dim, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())

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

                fc1 = ly.fully_connected(flat_1, 64, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())

                self.c_addition += (x_shape[-1]*64)
                self.c_multiplication += (x_shape[-1]*64)

                fc1 = tf.nn.dropout(fc1, keep_prob=self.keep_prob)

                x_shape = fc1.get_shape().as_list()

                fc2 = ly.fully_connected(fc1, 64, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())

                self.c_addition += (x_shape[-1]*64)
                self.c_multiplication += (x_shape[-1]*64)

                fc2 = tf.nn.dropout(fc2, keep_prob=self.keep_prob)
                for i in temp_5[0]:
                    with tf.variable_scope(i):

                        x_shape = fc2.get_shape().as_list()

                        output = ly.fully_connected(fc2, 2, activation_fn = None, weights_initializer=tf.contrib.layers.xavier_initializer())

                        self.c_addition += (x_shape[-1]*2)
                        self.c_multiplication += (x_shape[-1]*2)

                        prob = tf.nn.softmax(output)
                    logit.append(output)
                    prob_list.append(prob)

            with tf.variable_scope('group_2'):

                x_shape = flat_2.get_shape().as_list()

                fc1 = ly.fully_connected(flat_2, 64, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())

                self.c_addition += (x_shape[-1]*64)
                self.c_multiplication += (x_shape[-1]*64)

                fc1 = tf.nn.dropout(fc1, keep_prob=self.keep_prob)

                x_shape = fc1.get_shape().as_list()

                fc2 = ly.fully_connected(fc1, 64, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())

                self.c_addition += (x_shape[-1]*64)
                self.c_multiplication += (x_shape[-1]*64)

                fc2 = tf.nn.dropout(fc2, keep_prob=self.keep_prob)
                for i in temp_5[1]:
                    with tf.variable_scope(i):

                        x_shape = fc2.get_shape().as_list()

                        output = ly.fully_connected(fc1, 2, activation_fn = None, weights_initializer=tf.contrib.layers.xavier_initializer())

                        self.c_addition += (x_shape[-1]*2)
                        self.c_multiplication += (x_shape[-1]*2)

                        prob = tf.nn.softmax(output)
                    logit.append(output)
                    prob_list.append(prob)

            with tf.variable_scope('group_3'):      

                x_shape = flat_2.get_shape().as_list()

                fc1 = ly.fully_connected(flat_2, 64, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())

                self.c_addition += (x_shape[-1]*64)
                self.c_multiplication += (x_shape[-1]*64)

                fc1 = tf.nn.dropout(fc1, keep_prob=self.keep_prob)

                x_shape = fc1.get_shape().as_list()

                fc2 = ly.fully_connected(fc1, 64, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())

                self.c_addition += (x_shape[-1]*64)
                self.c_multiplication += (x_shape[-1]*64)

                fc2 = tf.nn.dropout(fc2, keep_prob=self.keep_prob)
                for i in temp_5[2]:
                    with tf.variable_scope(i):

                        x_shape = fc2.get_shape().as_list()

                        output = ly.fully_connected(fc1, 2, activation_fn = None, weights_initializer=tf.contrib.layers.xavier_initializer())

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
            latent = ly.fully_connected(final_latent, 5*5*128, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
            latent = self.bo_batch_norm(latent, self.is_train)
            latent = tf.reshape(latent, shape=[-1,5,5,128])       
            dim = 32
            latent = ly.conv2d_transpose(latent, dim*4, kernel_size=3, stride=2, padding='SAME', activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
            latent = tf.nn.relu(latent)
            upsample1 = ly.conv2d_transpose(latent, dim*4, kernel_size=3, stride=2, padding='VALID', activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
            upsample1 = tf.nn.relu(upsample1)
            upsample2 = ly.conv2d_transpose(upsample1, dim*2, kernel_size=3, stride=2, padding='VALID', activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
            upsample2 = tf.nn.relu(upsample2)
            upsample3 = ly.conv2d_transpose(upsample2, dim*1, kernel_size=3, stride=2, padding='VALID', activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
            upsample3 = tf.nn.relu(upsample3)
            upsample4 = ly.conv2d_transpose(upsample3 ,3, kernel_size=3, stride=2, padding='VALID', activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer())
        return upsample4


    def adversary_lrr(self, final_latent, reuse=False):
        with tf.variable_scope('adversary_lrr') as scope:  
            if reuse: 
                scope.reuse_variables()
            recontruction = ly.fully_connected(final_latent, 175*175*3, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = None)
        return tf.reshape(recontruction, shape=[-1, 175, 175, 3])


    def adversary_krr(self, kernel_map, reuse=False):
        with tf.variable_scope('adversary_krr') as scope:  
            if reuse: 
                scope.reuse_variables()
            recontruction = ly.fully_connected(kernel_map, 175*175*3, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = None)
        return tf.reshape(recontruction, shape=[-1, 175, 175, 3])

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
                print(left_pad)
                print(right_pad)
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
        #stride =  [1,1,1]
        #filter_size = [16,32,64]
        with tf.variable_scope('compressor') as scope:
            if reuse : 
                scope.reuse_variables()  
            #x = self._conv(image, [3, 3, 3, 16], 1)    
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

            #x = self.bo_batch_norm(x,self.is_train)
            #x = tf.reduce_mean(x,[1,2])  ### average pooling layer
            #x = ly.fully_connected(x,512,activation_fn=tf.nn.relu)  ### fix the latent dimension
            #x = tf.reshape(x,shape=[-1,32,32,3])
            return x 


    def RFF_map(self, input_tensor_1, input_tensor_2, seed, stddev, output_dim):
        input_tensor = tf.concat([input_tensor_1, input_tensor_2], axis=1)
        print("Information that the adversary can get: {}".format(input_tensor))
        random_state = check_random_state(seed)
        gamma = stddev
        omega_matrix_shape = [self.arg.dim*2, output_dim]
        bias_shape = [output_dim]

        '''
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
        self.lrr_mu_p = tf.placeholder(tf.float32, shape=[self.arg.dim*2])
        self.krr_mu_p = tf.placeholder(tf.float32, shape=[5000])
        self.t_mu_p = tf.placeholder(tf.float32, shape=[175*175*3])
        self.krr_weights = tf.placeholder(tf.float32, shape=[5000, 175*175*3])
        self.lrr_weights = tf.placeholder(tf.float32, shape=[self.arg.dim*2, 175*175*3])
        self.lrr_mu = self.init_tensor([self.arg.dim*2])
        self.krr_mu = self.init_tensor([5000])  
        self.t_mu = self.init_tensor([175*175*3])

        ## Center adjust or not.s
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

    def compute_acc_test(self):

        acc_list = []
        for j , k in self.next_batch(self.te_data, self.te_label, self.batch_size):
            b = k.shape[0]  
            penal = np.array([[5,1] for i in range(b)])
            #no = np.random.normal(size=(b,64,64,3))
            no = np.random.laplace(size=(b, 175, 175,3))
            pred = self.sess.run(self.prob,feed_dict={self.image_p:j.reshape(b,64,64,3),self.label_p:k,self.noise_p:no,self.keep_prob:1,self.penalty:penal})
            acc_list += list(np.argmax(pred,1))

        index_pos = []
        index_neg = []
        for i in range(len(self.te_label)):
            if self.te_label[i] ==1:
                index_pos.append(i)
            else : 
                index_neg.append(i)

        correct_pos = []
        correct_neg = []

        for i in index_pos : 
            if self.te_label[i] == acc_list[i]:
                correct_pos.append(1)

        for i in index_neg:
            if self.te_label[i] == acc_list[i]:
                correct_neg.append(1)

        a = len(correct_pos)/len(index_pos)
        b = len(correct_neg)/len(index_neg)
        print('{} is the mean accuracy : True Positive rate with True negative rate'.\
            format(1/2*(a+b)))

        return 1/2*(a+b)


    def plot(self,x):
        x = x - np.min(x)
        x =x /  np.max(x)
        x *= 255  
        x= x.astype(np.uint8)
        x = x.reshape(175, 175, 3)
        return x 


    def plot_10slot(self,name):

        r, c  = 2, 10
        j = np.array(self.preprocess(self.read(self.te_data[:128])))
        k = np.array([i for i in range(128)])
        penal = np.array([[0.5,1] for i in range(128)])
        no = np.random.laplace(size=(b, 175, 175,3))
        uu = self.sess.run(self.latent, feed_dict={self.image_p:j.reshape(128, 175, 175, 3),self.keep_prob:1,self.is_train:False})
        yy = self.sess.run(self.up, feed_dict={self.latent:uu, self.is_train:False})
        plt.figure(figsize=(10, 2))

        n = 10

        for i in range(n):

            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(self.plot(j[i]))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(self.plot(yy[i]))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.savefig('reconstructed_image'+str(name))
        plt.close()

    def resize(self,image):
        a = []
        for i in image : 
            a.append(self.preprocess(imresize(i, (175,175))))
        return a 

    def read(self,image):
        a = []
        for i in image : 
            a.append(imread(i))
        return a 

    def plot_175(self,x):

        x = x - np.min(x)
        x =x /  np.max(x)
        x *= 255  
        x= x.astype(np.uint8)
        x = x.reshape(175, 175, 3)

        return x 


    def compute_reco_mse(self, data, label):

        ##### after assign all the weights !!!!! 

        error_nn = []
        error_lrr = []
        error_krr = []
        for i , j in self.next_batch(data, label, self.batch_size):

            b = i.shape[0]
            no = np.random.normal(size=(b, 175, 175, 3))
            #no = np.random.laplace(size=(b, 175, 175,3))
            up_nn = self.sess.run(self.recon_nn, feed_dict={self.image_p:i.reshape(b, 175, 175, 3), self.noise_p:no, self.is_train:False})
            up_lrr = self.sess.run(self.recon_lrr, feed_dict={self.image_p:i.reshape(b, 175, 175, 3), self.noise_p:no, self.is_train:False})
            up_krr = self.sess.run(self.recon_krr, feed_dict={self.image_p:i.reshape(b, 175, 175, 3), self.noise_p:no, self.is_train:False})

            for k in range(len(up_nn)):
                #error.append(mean_squared_error(self.plot(i[k]).flatten(),self.plot(up[k]).flatten()))
                error_nn.append(mean_squared_error(i[k].flatten(), up_nn[k].flatten()))
                error_lrr.append(mean_squared_error(i[k].flatten(), up_lrr[k].flatten()))
                error_krr.append(mean_squared_error(i[k].flatten(), up_krr[k].flatten()))

        imsave('original.png', self.plot_175(i[0]))
        imsave('nn_reco.png', self.plot_175(up_nn[0]))
        imsave('lrr_reco.png', self.plot_175(up_lrr[0]))
        imsave('krr_reco.png', self.plot_175(up_krr[0]))

        return np.mean(error_nn), np.mean(error_lrr), np.mean(error_krr)

    def kernel_rbf(self, x, y): 
        b = len(self.t_data)
        K = euclidean_distances(x, y, squared=True)
        gamma = 1.0 / b
        K *= -(2*gamma)
        return np.exp(K)

    def kernel_matrix(self):

        emb_list = []
        for i,j in self.next_batch(self.t_data, self.t_label, self.batch_size):
            b = j.shape[0]
            penal = np.array([[0.5,1] for i in range(b)])
            no = np.random.normal(size=(128,175,175,3))
            uu = self.sess.run(self.latent, feed_dict={self.image_p:j.reshape(128, 175, 175, 3),self.keep_prob:1,self.is_train:False})
            emb_list.append(uu)

        count = 0
        for i in emb_list: 
            if count == 0: 
                emb_matrix = i 
                count +=1 
                continue
            emb_matrix = np.concatenate((emb_matrix,i),axis=0)

        D = np.zeros((len(self.t_data),len(self.t_data)))

        for i in range(len(self.t_data)):
            for j in range(i):
                #D[i,j]=quadraticChiDist(X[i,:],X[j,:])
                D[i,j]= self.kernel_rbf(emb_matrix[i,:], emb_matrix[j:])
                D[j,i]=D[i,j]
                ### or you can use the inner product of the matrix (mapping matrix (notated fi in Kung's textbook.))
        return D 


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
        print('Shape of KRR weights: {}.'.format(weights.shape))
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
        print('Shape of LRR weights: {}'.format(weights.shape))
        return weights, mu

    def get_emb_matrix(self): 

        count = 0
        for i,j in self.next_batch(self.t_data, self.t_label, self.batch_size):
            b = i.shape[0]
            penal = np.array([[0.5,1] for i in range(b)])
            no = np.random.normal(size=(128, 175, 175, 3))
            uu, yy = self.sess.run([self.latent_concat, self.kernel_map], feed_dict={self.image_p:i.reshape(b, 175, 175, 3), self.keep_prob:1, self.is_train:False})
            if count == 0 : 
                emb_matrix_lrr = uu
                emb_matrix_krr = yy 
                count+=1 
            else : 
                emb_matrix_lrr = np.concatenate((emb_matrix_lrr, uu), axis=0)
                emb_matrix_krr = np.concatenate((emb_matrix_krr, yy), axis=0)
                count+=1 
        print('Successfully')   
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

        error_nn, error_lrr, error_krr = self.compute_reco_mse(self.v_data, self.v_label)
        error_list.append(error_nn) 
        error_list.append(error_lrr)
        error_list.append(error_krr)
        print('Average MSE among all testing images is {}, {}, {}.(nn,lrr,krr)'.format(error_nn, error_lrr, error_krr))
        optimize_g = update_choice[np.argmin(error_list)]

        return optimize_g, feed_dict_assign


    def train(self):

        acc_trace = [] 
        mse_trace = [] 
        mse_trace_1 = []
        epochs = 15
        #os.mkdir('Male_2_test')
        #f = open('Male_2_test/Male_noise_log.txt','w')
        loss_trace = []

        train_matrix = self.get_train_matrix()
        train_mu = np.mean(train_matrix, axis=0)

        #update_choice = [self.g_opt_nn, self.g_opt_lrr, self.g_opt_krr]
        for i in range(epochs):
            citers = 15
            epoch_loss = []

            ### compute weights while assigning them  (old version may cost too many memory !!!)

            start = time.time()
            for j , k in self.train_next_batch(self.t_data, self.t_label, self.batch_size):

                b = j.shape[0]
                no = np.random.normal(size=(b, 175, 175,3))
                #no = np.random.laplace(size=(b, 175, 175,3))
                feed_dict = {}
                feed_dict[self.image_p] = j
                for attr in range(40):
                    feed_dict[self.label_list[attr]] = np.array(k[attr]).reshape(-1)
                feed_dict[self.keep_prob] = 1
                feed_dict[self.is_train] = True
                feed_dict[self.noise_p] = no

                #print("Training R")
                for _ in range(citers):
                    _ = self.sess.run(self.r_opt, feed_dict = feed_dict)

                c_loss , _ = self.sess.run([self.loss_c, self.c_opt], feed_dict = feed_dict)

            end = time.time()

            print("Training for R and C costs about {}.".format(end-start))

            start = time.time()

            optimize_g, feed_dict = self.assign(train_matrix, train_mu)
            
            for j , k in self.train_next_batch(self.t_data, self.t_label, self.batch_size):

                b = j.shape[0]
                no = np.random.normal(size=(b, 175, 175,3))
                #no = np.random.laplace(size=(b, 175, 175,3))
                #feed_dict = {}
                feed_dict[self.image_p] = j
                for attr in range(40):
                    feed_dict[self.label_list[attr]] = np.array(k[attr]).reshape(-1)
                feed_dict[self.keep_prob] = 1
                feed_dict[self.is_train] = True
                feed_dict[self.noise_p] = no
                for _ in range(1):
                    _  = self.sess.run([optimize_g], feed_dict=feed_dict)

            end = time.time()
            print("Training for G costs about {}.".format(end-start))

            av_acc = self.compute_acc(self.te_data, self.te_label, 'test')
            at_acc = self.compute_acc(self.v_data, self.v_label, 'validation')
            print('{}/{} epochs, cost {} sec, testing accuracy: {}'.format(i+1, epochs,end-start, av_acc))
            print('{} is the average accuracy among the 40 attributes (testing).'.format(np.mean(av_acc)))
            print('{} is the average accuracy among the 40 attributes (validation).'.format(np.mean(at_acc)))

            #if (i+1) % 2 == 0 :

            if self.arg.noise : 
                self.save_g()          
            else : 

                self.save_g()
            #np.save('double_emb/mse_trace.npy',mse_trace)
            #np.save('double_emb/mse_trace_1.npy',mse_trace_1)

            #self.plot_10slot(i+1)
            #if i == 14:
            #   self.saver.save(self.sess,'double_emb/model_multi')
            #f.write('Average of TPR and TNR is: '+str(tr_acc)+', average testing accuracy is : '+str(av_acc)+'\n')

            #self.saver.save(self.sess,'Male_2_test/model_'+str(i))    

        #f.close()

    def save_g(self):

        if self.arg.noise : 
            np.save('multi_adv/weights_'+str(self.arg.dim)+'_'+'noise'+'.npy',self.sess.run(self.theta_g))
        else : 
            np.save('multi_adv/weights_'+str(self.arg.dim)+'.npy',self.sess.run(self.theta_g))



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
        print('Shuffling needs about {} sce.'.format(end-start))

        return temp_img, temp_1
       

    def train_next_batch(self, img, label_list, batch_size):
        le = len(img)
        epo = le // batch_size
        leftover = le - (epo*batch_size)
        count = 0 
        '''
        c = list(zip(encoder_input,label))
        random.shuffle(c)
        temp , temp_1 = zip(*c)
        '''
        temp , temp_1 = self.shuffle()

        for i in range(0, le, batch_size):
            if i ==  (epo *batch_size) : 
                oo = [[] for i in range(40)]
                for k in range(40):
                    oo[k].append(temp_1[k][i:])    
                yield np.array(self.preprocess(self.read(temp[i:]))), oo
            else : 
                oo = [[] for i in range(40)]
                for k in range(40):
                    oo[k].append(temp_1[k][i: i+batch_size])
                yield np.array(self.preprocess(self.read(temp[i: i+batch_size]))), oo


    def next_batch(self, encoder_input, label, batch_size):
        le = len(encoder_input)
        epo = le//batch_size
        leftover = le - (epo*batch_size)
        for i in range(0, le, batch_size):
            if i ==  (epo *batch_size) : 
                oo = [[] for i in range(40)]
                for k in range(40):
                    oo[k].append(label[k][i:]) 

                yield np.array(self.preprocess(self.read(encoder_input[i:]))), oo  #np.array(label[i:])

            else : 
                oo = [[] for i in range(40)]
                for k in range(40):
                    oo[k].append(label[k][i: i+batch_size])  

                yield np.array(self.preprocess(self.read(encoder_input[i: i+batch_size]))), oo  #np.array(label[i:i+256])


    def compute_acc(self, te_data, te_label, name):
        
        acc_list = []
        pred_list = [[] for i in range(40)]
        for j , k in self.next_batch(te_data, te_label, self.batch_size):
            b = j.shape[0]
            no = np.random.normal(size=(b, 175, 175, 3))
            #no = np.random.laplace(size=(b, 175, 175,3))
            feed_dict = {}
            feed_dict[self.image_p] = j.reshape(b, 175, 175, 3)
            feed_dict[self.is_train] = False
            feed_dict[self.keep_prob] = 1
            feed_dict[self.noise_p] = no
            temp = self.sess.run(self.prob_list, feed_dict=feed_dict)
            for i in range(40):
                pred_list[i].append(temp[i])

        for i in range(40):
            temp = np.concatenate(pred_list[i], axis=0)
            acc = accuracy_score(np.argmax(temp,1), te_label[i])
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

    def get_nb_params_shape(self,shape):
        '''
        Computes the total number of params for a given shap.
        Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
        '''
        nb_params = 1
        for dim in shape:
            nb_params = nb_params*int(dim)
        return nb_params 

    def test(self):

        if self.arg.noise : 
            self.saver.restore(self.sess, self.arg.path_mdoel)
        else : 
            self.saver.restore(self.sess,self.arg.path_mdoel)

        av_acc = self.compute_acc(self.te_data,self.te_label,'test')
        print('Testing accuracy: {}'.format(av_acc))
        print(np.mean(av_acc))


    ## More comment .... (MNIST and HAR must be finished this week ....)



