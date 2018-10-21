import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import h5py 
from functools import reduce
import tensorflow as tf
import tensorflow.contrib.layers as ly 
import time
from scipy.misc import imread, imresize ,imsave
from sklearn.model_selection import train_test_split
import random
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

plt.switch_backend('agg')
tf.set_random_seed(9)
class cpgan: 
    def __init__(self,args):
        self.arg = args
        #all_data , all_label = self.load_data()
        self.t_data ,self.te_data ,self.t_label ,self.te_label = self.load_data()
        print(len(self.t_label))
        print(len(self.t_label[0]))
        assert len(self.t_label[0]) == len(self.t_label[1])

        sam = []
        #for i in all_label:
        #    if i == 1: 
        #        sam.append(i)
        #if len(sam) < 50000 : 
        #    self.t_data ,self.te_data ,self.t_label ,self.te_label = self.sample(all_data,all_label)
        #else : 
        #    self.t_data ,self.te_data ,self.t_label ,self.te_label = train_test_split(all_data,all_label,test_size=0.05,random_state=9)
        
        #self.t_data ,self.te_data ,self.t_label ,self.te_label = train_test_split(all_data,all_label, test_size=0.2, random_state=9)

        self.build_model()

        qq = self.count_number_trainable_params()

        print('Total parameters of this model is {}'.format(qq))
        #self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob,1),self.label_p),tf.float32))

        '''
        classes = [9, 10, 5, 2, 4, 5, 3, 2] 
        utility_loss_list = []
        for i in range(40):
            #utility_loss_list.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label_list[i],\
            #                                                                               logits=self.logit_list[i])))
            utility_loss_list.append(tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot_list[i],\
                                                                             logits=self.logit_list[i]))
        '''


        #utility_loss = tf.reduce_mean(tf.add_n(utility_loss_list))

        self.loss_r_1 = tf.losses.mean_squared_error(self.image_p,self.up) 
        self.loss_r_2 = tf.losses.mean_squared_error(self.image_p,self.up_1) 

        #self.loss_c = utility_loss

        ## only concern about the uitlity classification accuracy.

        #self.loss_g = utility_loss - tf.losses.mean_squared_error(self.image_p,self.up) 

        self.theta_r_1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='reconstructor_1')
        self.theta_r_2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='reconstructor_2')
        self.theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='compressor')

        #self.theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='utility_classifier')

        #print(self.theta_c)

        uti_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(uti_update):

            #self.c_op = tf.train.AdamOptimizer(0.001)
            #self.c_opt = self.c_op.minimize(self.loss_c, var_list = self.theta_g + self.theta_c)

            #self.g_op = tf.train.AdamOptimizer(0.001)
            #self.g_opt = self.g_op.minimize(self.loss_g,var_list=self.theta_g)

            #self.c_op = tf.train.AdamOptimizer(0.001)
            #self.c_opt = self.c_op.minimize(self.loss_c,var_list=self.theta_c)

            self.r_op_1 = tf.train.AdamOptimizer(0.001)
            self.r_opt_1 = self.r_op_1.minimize(self.loss_r_1, var_list=self.theta_r_1)


            self.r_op_2 = tf.train.AdamOptimizer(0.001)
            self.r_opt_2 = self.r_op_2.minimize(self.loss_r_2, var_list=self.theta_r_2)

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=15)

        #a = np.load('double_emb/weights_1728.npy')
        #a = np.load('double_emb/weights_64_gaussian_noise.npy')
        a = np.load('multcpagn_log/theta_g.npy')
        '''
        self.assign_op = []
        for i in range(len(self.theta_g)):
            print(a[i].shape)
            if a[i].shape[0] > 3 :
                self.assign_op.append(tf.assign(self.theta_g[i],a[i].reshape(3,3,3,3)))
            else : 
                self.assign_op.append(tf.assign(self.theta_g[i],a[i]))

            #self.assign_op.append(tf.assign(self.theta_g[i],temp[i]))          
        '''

        self.assign_op = []
        for i in range(len(self.theta_g)):
            #print(a[i].shape)
            self.assign_op.append(tf.assign(self.theta_g[i],a[i]))

            #self.assign_op.append(tf.assign(self.theta_g[i],temp[i]))  

        print('Length of all parameters in utility_classifier should be assigned:{}'.format(len(self.assign_op)))

        
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
        #classes = [9, 10, 5, 2, 4, 5, 3, 2] 

        temp =list(np.sort(os.listdir(self.arg.path)))
        #print(temp)
        indices = [i for i in range(len(temp))]
        img = []
        count = 0
        for i in temp : 
            #im = imread(os.path.join(self.arg.path,i))
            #im = imresize(imread(os.path.join(self.arg.path,i)),size=(64,64))
            #im = imresize(imread(os.path.join(self.arg.path,i)),size=(256,256))
            #im = (im/127.5)-1
            #img.append(im)
            img.append(os.path.join(self.arg.path,i))

        t_img ,te_img ,train_indices ,test_indices = train_test_split(img, indices, test_size=0.09, random_state=9, shuffle=False)

        #40_lab = dict()
        lab_40_t = []
        lab_40_val = [] 

        start = time.time()
        #for i in range(8):
        #   a = 'subnet_'+str(i+1)
        pp = 0
        for j in temp_5: 
            for q in j :
                #print(pp)
                lab_t = []
                lab_v = []
                temp = pd.read_csv('/home/pywu/bowei/All_label.csv')[[q]].values.reshape(-1)
                temp = np.array(temp)
                tra = temp[train_indices]
                tes = temp[test_indices]

                tra[tra==-1] = 0
                tes[tes==-1] = 0

                '''
                for i in range(len(temp)) : 
                    qq = temp[i]
                    if i in train_indices:
                        if qq == -1:
                            lab_t.append(0)
                        else : 
                            lab_t.append(1)
                    else : 
                        if qq == -1:
                            lab_v.append(0)
                        else : 
                            lab_v.append(1)  
                '''                   
                #lab_40_t.append(lab_t)
                #lab_40_val.append(lab_v)
                lab_40_t.append(list(tra))
                lab_40_val.append(list(tes))
                pp+=1

        end = time.time()
        print('Load successfully!!! And cost about {} sec'.format(end-start))

        return t_img, te_img, lab_40_t, lab_40_val

    def preprocess(self,data):
        a=[]
        for i in  data: 
            #temp = imresize(i,(175,175))
            temp = self.plot(i)
            temp = imresize(i,(175,175))
            temp = (temp/127.5)-1
            a.append(temp)
        return a

    def bo_batch_norm(self,x, is_training, momentum=0.9, epsilon=0.0001):
        x = tf.layers.batch_normalization(x, momentum=momentum, epsilon=epsilon,training=is_training)
        return x

    def shallow_generator_conv(self, image, reuse=False):
        dim = 64
        with tf.variable_scope('compressor') as scope:
            if reuse : 
                scope.reuse_variables()
            conv1 = ly.conv2d(image,dim*1,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.leaky_relu,weights_initializer=tf.random_normal_initializer(0, 0.02))
            conv1 = self.bo_batch_norm(conv1, self.is_train)
            conv1 = ly.max_pool2d(conv1,kernel_size=3,stride=2,padding='SAME')

            conv2 = ly.conv2d(conv1,dim*2,kernel_size=3,stride=2,padding='SAME',activation_fn=tf.nn.leaky_relu,weights_initializer=tf.random_normal_initializer(0, 0.02))
            conv2 = self.bo_batch_norm(conv2, self.is_train)

            conv3 = ly.conv2d(conv2,dim*4,kernel_size=3,stride=2,padding='SAME',activation_fn=tf.nn.leaky_relu,weights_initializer=tf.random_normal_initializer(0, 0.02))
            conv3 = self.bo_batch_norm(conv3, self.is_train)

            conv4 = ly.conv2d(conv3,dim*8,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.leaky_relu,weights_initializer=tf.random_normal_initializer(0, 0.02))
            conv4 = self.bo_batch_norm(conv4, self.is_train)
            conv4 = ly.max_pool2d(conv4,kernel_size=3,stride=2,padding='SAME')
            #latent = ly.fully_connected(tf.reshape(conv4,shape=[-1,4*4*dim*8]),4096,activation_fn=tf.nn.leaky_relu)
            latent = ly.fully_connected(tf.reshape(conv4,shape=[-1,7*7*dim*8]),512,activation_fn=tf.nn.leaky_relu)
            #print(conv4)
        return latent 

    def generator_conv(self,image,reuse=False):
        with tf.variable_scope('compressor') as scope:
            if reuse:
                scope.reuse_variables()  
            '''
            conv1 = ly.conv2d(image,96,kernel_size=11,stride=4,padding='SAME',activation_fn=tf.nn.relu,weights_initializer=tf.random_normal_initializer(0, 0.02))
            conv1 = self.bo_batch_norm(conv1, self.is_train)
            conv1 = ly.max_pool2d(conv1,kernel_size=3,stride=2,padding='SAME')

            conv2 = ly.conv2d(conv1,256,kernel_size=5,stride=1,padding='SAME',activation_fn=tf.nn.relu,weights_initializer=tf.random_normal_initializer(0, 0.02))
            conv2 = self.bo_batch_norm(conv2, self.is_train)
            conv2 = ly.max_pool2d(conv2,kernel_size=3,stride=2,padding='SAME')

            conv3 = ly.conv2d(conv2,384,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.relu,weights_initializer=tf.random_normal_initializer(0, 0.02))
            conv3 = self.bo_batch_norm(conv3, self.is_train)
            #conv3 = ly.max_pool2d(conv3,kernel_size=3,stride=2,padding='SAME')

            conv4 = ly.conv2d(conv3,384,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.relu,weights_initializer=tf.random_normal_initializer(0, 0.02))
            conv4 = self.bo_batch_norm(conv4, self.is_train)
            #conv4 = ly.max_pool2d(conv4,kernel_size=3,stride=2,padding='SAME')

            conv5 = ly.conv2d(conv4,256,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.relu,weights_initializer=tf.random_normal_initializer(0, 0.02))
            conv5 = self.bo_batch_norm(conv5, self.is_train)
            conv5 = ly.max_pool2d(conv5,kernel_size=3,stride=2,padding='SAME')

            flat = ly.flatten(conv5)
            fc1 = ly.fully_connected(flat, 4096, activation_fn=tf.nn.relu)
            fc2 = ly.fully_connected(fc1, 4096, activation_fn=tf.nn.relu)
            '''
            conv1 = ly.conv2d(image, 96, kernel_size=3, padding='VALID', stride=2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            conv1 = ly.max_pool2d(conv1, kernel_size=3, padding='VALID', stride=2)

            conv2 = ly.conv2d(conv1, 256, kernel_size=5, padding='SAME', stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            conv2 = ly.max_pool2d(conv2, kernel_size=3, padding='SAME', stride=2)

            with tf.variable_scope('subenet_first_group'):
                with tf.variable_scope('branch_1'):
                    pool_1 = ly.max_pool2d(conv2, kernel_size=3, padding='SAME', stride=2)
                    conv_1 = ly.conv2d(pool_1, 64, kernel_size=3, padding='SAME',stride=2, weights_initializer=tf.contrib.layers.xavier_initializer())

                with tf.variable_scope('branch_2'):
                    reduce_3 = ly.conv2d(conv2, 64, kernel_size=1, padding='SAME', stride=2, weights_initializer=tf.contrib.layers.xavier_initializer())
                    conv_3 = ly.conv2d(reduce_3, 64, kernel_size=3, padding='SAME', stride=2, weights_initializer=tf.contrib.layers.xavier_initializer())

                with tf.variable_scope('branch_3'):
                    reduce_5 = ly.conv2d(conv2, 64, kernel_size=1, padding='SAME', stride=2, weights_initializer=tf.contrib.layers.xavier_initializer())
                    conv_5 = ly.conv2d(reduce_5, 64, kernel_size=3, padding='SAME', stride=2, weights_initializer=tf.contrib.layers.xavier_initializer())

                output = tf.concat([conv_1,conv_3,conv_5],axis=2)
                pool3_a = ly.max_pool2d(output,kernel_size=3, padding='SAME', stride=2)
                flat_first = ly.flatten(pool3_a)
                flat_first = ly.fully_connected(flat_first, 64, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())


            with tf.variable_scope('subnet_second_group'):
                with tf.variable_scope('branch_1'):
                    pool_1 = ly.max_pool2d(conv2, kernel_size=3, padding='SAME', stride=2)
                    conv_1 = ly.conv2d(pool_1, 64, kernel_size=3, padding='SAME',stride=2, weights_initializer=tf.contrib.layers.xavier_initializer())

                with tf.variable_scope('branch_2'):
                    reduce_3 = ly.conv2d(conv2, 64, kernel_size=1, padding='SAME', stride=2, weights_initializer=tf.contrib.layers.xavier_initializer())
                    conv_3 = ly.conv2d(reduce_3, 64, kernel_size=3, padding='SAME', stride=2, weights_initializer=tf.contrib.layers.xavier_initializer())

                with tf.variable_scope('branch_3'):
                    reduce_5 = ly.conv2d(conv2, 64, kernel_size=1, padding='SAME', stride=2, weights_initializer=tf.contrib.layers.xavier_initializer())
                    conv_5 = ly.conv2d(reduce_5, 64, kernel_size=3, padding='SAME', stride=2, weights_initializer=tf.contrib.layers.xavier_initializer())
                output = tf.concat([conv_1, conv_3, conv_5],axis=2)
                pool3_a = ly.max_pool2d(output,kernel_size=3, padding='SAME', stride=2)
                flat_second = ly.flatten(pool3_a)
                flat_second = ly.fully_connected(flat_second, 64, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())

            return flat_first , flat_second

    def utility_classifier(self,flat_1,flat_2,reuse=False):

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

        logit = []
        prob_list = []
        with tf.variable_scope('utility_classifier'):

            with tf.variable_scope('group_1'):
                fc1 = ly.fully_connected(flat_1, 64, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
                fc1 = tf.nn.dropout(fc1, keep_prob=self.keep_prob)
                fc2 = ly.fully_connected(fc1, 64, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
                fc2 = tf.nn.dropout(fc2, keep_prob=self.keep_prob)
                for i in temp_5[0]:
                    with tf.variable_scope(i):
                        output = ly.fully_connected(fc1, 2, activation_fn = None, weights_initializer=tf.contrib.layers.xavier_initializer())
                        prob = tf.nn.softmax(output)
                    logit.append(output)
                    prob_list.append(prob)

            with tf.variable_scope('group_2'):
                fc1 = ly.fully_connected(flat_2, 64, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
                fc1 = tf.nn.dropout(fc1, keep_prob=self.keep_prob)
                fc2 = ly.fully_connected(fc1, 64, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
                fc2 = tf.nn.dropout(fc2, keep_prob=self.keep_prob)
                for i in temp_5[1]:
                    with tf.variable_scope(i):
                        output = ly.fully_connected(fc1, 2, activation_fn = None, weights_initializer=tf.contrib.layers.xavier_initializer())
                        prob = tf.nn.softmax(output)
                    logit.append(output)
                    prob_list.append(prob)

            with tf.variable_scope('group_3'):      
                fc1 = ly.fully_connected(flat_2, 64, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
                fc1 = tf.nn.dropout(fc1, keep_prob=self.keep_prob)
                fc2 = ly.fully_connected(fc1, 64, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
                fc2 = tf.nn.dropout(fc2, keep_prob=self.keep_prob)
                for i in temp_5[2]:
                    with tf.variable_scope(i):
                        output = ly.fully_connected(fc1, 2, activation_fn = None, weights_initializer=tf.contrib.layers.xavier_initializer())
                        prob = tf.nn.softmax(output)
                    logit.append(output)
                    prob_list.append(prob)

        return logit , prob_list

    def decoder_conv(self, latent, name, reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            '''
            dim = 32       
            upsample1 = ly.conv2d_transpose(latent,dim*4,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.leaky_relu,weights_initializer=tf.random_normal_initializer(0, 0.02))
            upsample1 = self.bo_batch_norm(upsample1, self.is_train)
            upsample2 = ly.conv2d_transpose(upsample1, dim*2,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.leaky_relu,weights_initializer=tf.random_normal_initializer(0, 0.02))
            upsample2 = self.bo_batch_norm(upsample2, self.is_train)
            upsample5 = ly.conv2d_transpose(upsample2, dim*1,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.leaky_relu,weights_initializer=tf.random_normal_initializer(0, 0.02))
            upsample5 = self.bo_batch_norm(upsample5, self.is_train)
            upsample6 = ly.conv2d_transpose(upsample5 ,3,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.tanh,weights_initializer=tf.random_normal_initializer(0, 0.02))
            '''
            latent = ly.fully_connected(latent,5*5*64, activation_fn =tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))#,normalizer_fn=ly.batch_norm)
            latent = self.bo_batch_norm(latent, self.is_train)
            latent = tf.reshape(latent,shape=[-1,5,5,64])
            dim = 32
            print(latent)
            latent = ly.conv2d_transpose(latent, dim*4,kernel_size=3,stride=2,padding='SAME',activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
            latent = self.bo_batch_norm(latent, self.is_train)
            upsample1 = ly.conv2d_transpose(latent, dim*4,kernel_size=3,stride=2,padding='VALID',activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
            upsample1 = self.bo_batch_norm(upsample1, self.is_train)
            upsample2 = ly.conv2d_transpose(upsample1, dim*2,kernel_size=3,stride=2,padding='VALID',activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
            upsample2 = self.bo_batch_norm(upsample2, self.is_train)
            upsample5 = ly.conv2d_transpose(upsample2, dim*1,kernel_size=3,stride=2,padding='VALID',activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
            upsample5 = self.bo_batch_norm(upsample5, self.is_train)
            upsample6 = ly.conv2d_transpose(upsample5, 3, kernel_size=3,stride=2,padding='VALID',activation_fn=tf.nn.tanh, weights_initializer=tf.random_normal_initializer(0, 0.02))
            print(upsample6)
        return upsample6  

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

    def build_model(self):

        tf.reset_default_graph()

        #self.image_p = tf.placeholder(tf.float32,shape=(None,256,256,3))

        self.image_p = tf.placeholder(tf.float32,shape=(None,175, 175, 3))
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

        '''
        for i in range(40):
            self.label_+str(i) = tf.placeholder(tf.int64,shape=(None)) 
            self.label_list.append(self.label_+str(i))
        '''
        self.noise_p = tf.placeholder(tf.float32,shape=(None,175, 175, 3))
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_train = tf.placeholder(tf.bool)
        self.noisy_img = tf.add(self.image_p,self.noise_p)
        #self.latent ,self.latent_1 = self.generator_conv(self.image_p)
        self.latent ,self.latent_1 = self.generator_conv(self.noisy_img)


        #self.latent = self.residual_g(self.image_p)
        #self.latent = self.generator_conv(self.noisy_img)
        #self.latent = self.shallow_generator_conv(self.image_p)


        print(self.latent)
        print(self.latent_1)
        #self.up = self.decoder_conv(self.latent_no,reuse=False)


        #self.emb = tf.concat([self.latent,self.latent_1],axis=1)
        #self.up = self.decoder_conv(self.emb)
        self.up = self.decoder_conv(self.latent,'reconstructor_1')
        self.up_1 = self.decoder_conv(self.latent_1,'reconstructor_2')


        print(self.up)
        print(self.up_1)


    def compute_acc_test(self):
        acc_list = []
        for j , k in self.next_batch(self.te_data,self.te_label):
            b = k.shape[0]
            penal = np.array([[5,1] for i in range(b)])
            no = np.random.normal(size=(b,64,64,3))
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
        print(a)
        print(b)
        print('{} is the mean accuracy : True Positive rate with True negative rate'.\
            format(1/2*(a+b)))

        return 1/2*(a+b)


    def plot(self,x):

        x = x - np.min(x)

        x =x /  np.max(x)

        x *= 255 
        x = x.astype(np.uint8)
        x = x.reshape(112, 112, 3)
        return x 

    def plot_1(self,x):
        x = x - np.min(x)
        x = x /  np.max(x)
        x *= 255 
        x = x.astype(np.uint8)
        x = x.reshape(175, 175, 3)
        return x 

    def plot_10slot(self,name):

        r ,c  = 3,10

        j = np.array(self.preprocess(self.read(self.te_data[:128])))
        k = np.array([i for i in range(128)])
        penal = np.array([[0.5,1] for i in range(128)])
        #no = np.random.laplace(size=(128, 175, 175,3))
        no = np.random.normal(size=(128, 175, 175,3))

        uu, kk = self.sess.run([self.latent,self.latent_1], feed_dict={self.noise_p:no, self.image_p:j.reshape(128, 175, 175, 3), self.keep_prob:1,self.is_train:False})
        yy, ll = self.sess.run([self.up, self.up_1], feed_dict={self.latent:uu, self.latent_1:kk, self.is_train:False})

        '''
        fig, axs = plt.subplots(r, c)
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        cnt = 0

        for i in range(r):
            for q in range(c):
                if i == 0 : 
                    #axs[i,j].imshow(self.plot_1(self.te_data[j]))
                    axs[i,q].imshow(self.plot_1(j[q]))
                    axs[i,q].axis('off')

                if i ==1 : 
                    axs[i,q].imshow(self.plot_1(yy[q]))
                    axs[i,q].axis('off')

                if i == 2 : 
                    axs[i,q].imshow(self.plot_1(ll[q]))
                    axs[i,q].axis('off')                
                #cnt += 1
        fig.savefig('double_emb/reconstructed_image'+str(name))
        plt.close()
        '''
        w=10
        h=10
        fig=plt.figure(figsize=(10, 3))
        columns = 10
        rows = 3
        for i in range(1, columns*rows +1): 
            if i < 11 : 
                #img = np.random.randint(10, size=(h,w))
                fig.add_subplot(rows, columns, i)
                plt.imshow(self.plot_1(j[i-1]))
                plt.axis('off')
            elif 11 <= i < 21 : 
                fig.add_subplot(rows, columns, i)
                plt.imshow(self.plot_1(yy[i-11]))
                plt.axis('off')
            else : 
                fig.add_subplot(rows, columns, i)
                plt.imshow(self.plot_1(ll[i-21]))
                plt.axis('off')
        #plt.show()
        fig.savefig('multcpagn_log/reconstructed_image_noise'+str(name))
        plt.close()
        
    def resize(self,image):
        a = []
        for i in image : 
            a.append(self.preprocess(imresize(i,(175,175))))
        return a 

    def read(self,image):
        a = []
        for i in image : 
            a.append(imread(i))
        return a 

    def compute_reco_mse(self):
        error = []
        error_1 = []
        for i , j in self.next_batch(self.te_data,self.te_label):
            b = i.shape[0]
            no = np.random.laplace(size=(b, 175, 175,3))
            up, up_1 = self.sess.run([self.up,self.up_1],feed_dict={self.noise_p:no, self.image_p:i.reshape(b, 175, 175,3), self.noise_p:no, self.is_train:False})
            for k in range(len(up)):
                #error.append(mean_squared_error(self.plot(i[k]).flatten(),self.plot(up[k]).flatten()))
                error.append(mean_squared_error(i[k].flatten(), up[k].flatten()))
                error_1.append(mean_squared_error(i[k].flatten(), up_1[k].flatten()))

        print('Average MSE among all testing images is {}.(emb_1)'.format(np.mean(error)))
        print('Average MSE among all testing images is {}.(emb_2)'.format(np.mean(error_1)))

        return np.mean(error) , np.mean(error_1)


    def train(self):
        acc_trace = [] 
        mse_trace = [] 
        mse_trace_1 = []
        epochs = 100
        #os.mkdir('Male_2_test')
        #f = open('Male_2_test/Male_noise_log.txt','w')
        loss_trace = []

        for i in range(epochs):
            #self.plot_10slot(i+1)
            #mse_1 , mse_2 = self.compute_reco_mse()

            #self.plot_10slot(i+1)

            citers = 1

            epoch_loss = []

            start = time.time()

            for j , k in self.train_next_batch(self.t_data,self.t_label):
                self.sess.run(self.assign_op)
                b = j.shape[0]
                #no = np.random.laplace(size=(b, 175, 175,3))
                no = np.random.normal(size=(b, 175, 175,3))
                feed_dict = {}
                feed_dict[self.image_p] = j
                for attr in range(40):
                    feed_dict[self.label_list[attr]] = np.array(k[attr]).reshape(-1)
                feed_dict[self.keep_prob] = 1
                feed_dict[self.is_train] = True
                feed_dict[self.noise_p] = no
                #for _ in range(citers):
                d_loss, d_loss_1, _ , _ = self.sess.run([self.loss_r_1, self.loss_r_2, self.r_opt_1,self.r_opt_2], feed_dict = feed_dict)

                #c_loss , _= self.sess.run([self.loss_c, self.c_opt], feed_dict = feed_dict)

                #for _ in range(3):
                #    _, lo = self.sess.run([self.g_opt, self.loss_r], feed_dict = feed_dict)
                
            end = time.time()

            print('Batch training loss is {} and {}.'.format(d_loss,d_loss_1))
            #print('{}/{} epochs , cost {} sec , the utility_loss = {}. reconstruction loss:{}'.format(i+1,epochs,end-start,c_loss,lo))

            #tr_acc = self.compute_acc_test()

            #av_acc = self.compute_acc(self.te_data,self.te_label,'test')

            #at_acc = self.compute_acc(self.t_data,self.t_label,'test')

            #print('{}/{} epochs, cost {} sec, testing accuracy: {}'.format(i+1, epochs,end-start, av_acc))

            #print('{} is the average accuracy of the 40 attributes (testing).'.format(np.mean(av_acc)))

            #if (i+1) % 5 == 0 :

            #self.saver.save(self.sess,'double_emb/model_reco')
            mse_1 , mse_2 = self.compute_reco_mse()
            mse_trace.append(mse_1)
            mse_trace_1.append(mse_2)
            np.save('multcpagn_log/mse_trace.npy',mse_trace)
            np.save('multcpagn_log/mse_trace_1.npy',mse_trace_1)
            if i%2==0:
                self.plot_10slot('4')
            #f.write('Average of TPR and TNR is: '+str(tr_acc)+', average testing accuracy is : '+str(av_acc)+'\n')
            #self.saver.save(self.sess,'Male_2_test/model_'+str(i))    
        #f.close()

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

        '''
        for i in range(len(self.t_data)):
            temp_img.append(temp[i][0])
        count = 1
        for i in temp_1:
            for k in range(len(self.t_data)):
                #temp_img.append(temp[k][0])
                i.append(temp[k][count])
            count+=1
        '''

        count = 1
        for i in range(len(self.t_data)):
            temp_img.append(temp[i][0])
            for k in range(1,41):
                temp_1[k-1].append(temp[i][k])


        end = time.time()
        print('Shuffling needs about {} sce.'.format(end-start))
        #print(len(temp_img))
        #print(len(temp_1))
        #print(len(temp_1[0]))

        return temp_img,temp_1
       
    def train_next_batch(self, img, label_list, batch_size=64):
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

        for i in range(0,le,64):
            if i ==  (epo *batch_size) : 
                oo = [[] for i in range(40)]
                for k in range(40):
                    oo[k].append(temp_1[k][i:])    
                yield np.array(self.preprocess(self.read(temp[i:]))), oo
            else : 
                oo = [[] for i in range(40)]
                for k in range(40):
                    oo[k].append(temp_1[k][i:i+64])
                yield np.array(self.preprocess(self.read(temp[i:i+64]))), oo

    def next_batch(self,encoder_input,label,batch_size=64):
        le = len(encoder_input)
        epo = le//batch_size
        leftover = le - (epo*batch_size)
        for i in range(0,le,64):
            if i ==  (epo *batch_size) : 
                oo = [[] for i in range(40)]
                for k in range(40):
                    oo[k].append(label[k][i:]) 

                yield np.array(self.preprocess(self.read(encoder_input[i:]))), oo  #np.array(label[i:])

            else : 
                oo = [[] for i in range(40)]
                for k in range(40):
                    oo[k].append(label[k][i:i+64])  

                yield np.array(self.preprocess(self.read(encoder_input[i:i+64]))), oo  #np.array(label[i:i+256])

    def compute_acc(self,te_data,te_label,name):
        acc_list = []
        pred_list = [[] for i in range(40)]
        for j , k in self.next_batch(te_data,te_label):
            b = j.shape[0]
            no = np.random.normal(size=(b,175, 175, 3))
            feed_dict = {}
            feed_dict[self.image_p] = j.reshape(b,175, 175, 3)
            feed_dict[self.is_train] = False
            feed_dict[self.keep_prob] = 1
            feed_dict[self.noise_p] = no
            temp = self.sess.run(self.prob_list,feed_dict=feed_dict)
            for i in range(40):
                pred_list[i].append(temp[i])

        for i in range(40):
            temp = np.concatenate(pred_list[i], axis=0)
            #temp[temp>0.5] = 1 
            #temp[temp<0.5] = 0
            #acc = accuracy_score(np.argmax(y_true,1),np.argmax(se,1))
            acc = accuracy_score(np.argmax(temp,1),te_label[i])
            #acc = accuracy_score(temp,te_label[i])
            acc_list.append(acc)

        plt.plot(acc_list)
        plt.xlabel('attribute index')
        plt.ylabel('accuracy')
        plt.savefig('Accuracy_of_each_attribute_'+name+'.png')
        return acc_list


    def count_number_trainable_params(self):
        '''
        Counts the number of trainable variables.
        '''
        tot_nb_params = 0
        for trainable_variable in tf.trainable_variables():
            shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
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
        temp = np.load('multcpagn_log/mse_trace.npy')
        #print(temp)
        le = len(list(temp))
        le = [i for i in range(le)]
        plt.plot(le,temp)
        plt.title('Training curve of the attcker network(multi_celebA)')
        #plt.yticks([0.5,0.6,0.7,0.8,0.9,1,1.1,1.2])
        plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2])
        #plt.yticks([0.6,0.7,0.8,0.9,1])
        #plt.yticks([0.2,0.22,0.24,0.26,0.28])
        plt.xlabel('Epochs')
        plt.ylabel('Mean square error')
        plt.savefig('Multi_celebA_mse_trace.png')






