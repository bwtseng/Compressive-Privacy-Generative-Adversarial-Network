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
import math 

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

        self.init_learning_rate = 0.001

        qq = self.count_number_trainable_params()

        print('Total parameters of this model is {}'.format(qq))
        #self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob,1),self.label_p),tf.float32))

        classes = [9, 10, 5, 2, 4, 5, 3, 2] 
        utility_loss_list = []
        for i in range(40):
            #utility_loss_list.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label_list[i],\
            #                                                                               logits=self.logit_list[i])))
            utility_loss_list.append(tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot_list[i],\
                                                                                         logits=self.logit_list[i]))
        #print(utility_loss_list[0])

        utility_loss = tf.reduce_mean(tf.add_n(utility_loss_list))

        #self.loss_r = tf.losses.mean_squared_error(self.image_p,self.up) 

        self.loss_c = utility_loss

        ## only concern about the uitlity classification accuracy.

        #self.loss_g = utility_loss - tf.losses.mean_squared_error(self.image_p,self.up) 

        #self.theta_r = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='reconstructor')

        self.theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='compressor')

        self.theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='utility_classifier')

        #print(self.theta_c)

        uti_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(uti_update):
            self.c_op = tf.train.MomentumOptimizer(self.learning_rate_p,0.9, use_nesterov=True)
            self.c_opt = self.c_op.minimize(self.loss_c,var_list=self.theta_c+self.theta_g)
            '''
            self.g_op = tf.train.MomentumOptimizer(self.learning_rate_p,0.9, use_nesterov=True)
            self.g_opt = self.g_op.minimize(self.loss_g,var_list=self.theta_g)

            self.r_op = tf.train.MomentumOptimizer(self.learning_rate_p,0.9, use_nesterov=True)
            self.r_opt = self.r_op.minimize(self.loss_r,var_list=self.theta_r)

            self.c_op = tf.train.MomentumOptimizer(self.learning_rate_p,0.9, use_nesterov=True)
            self.c_opt = self.c_op.minimize(self.loss_c,var_list=self.theta_c)
            '''

            '''
            self.g_op = tf.train.AdamOptimizer(0.001)
            self.g_opt = self.g_op.minimize(self.loss_g,var_list=self.theta_g)

            self.c_op = tf.train.AdamOptimizer(0.001)
            self.c_opt = self.c_op.minimize(self.loss_c,var_list=self.theta_c)

            self.r_op = tf.train.AdamOptimizer(0.001)
            self.r_opt = self.r_op.minimize(self.loss_r,var_list=self.theta_r)
            '''
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=15)
        
        #a = np.load('/home/pywu/bowei/fcn/weights.npy')
        a = np.load('/home/pywu/bowei/multi-celeba/pretrain/weights.npy')
        self.assign_op = []
        for i in range(len(self.theta_g)):
            self.assign_op.append(tf.assign(self.theta_g[i],a[i]))
        
    def load_data(self):
        
        temp_5 = []
        subnet_1 = ['Attractive', 'Blurry', 'Chubby', 'Heavy_Makeup', 'Male', 'Oval_Face', 'Pale_Skin','Smiling','Young']
        temp_5.append(subnet_1)
        subnet_2 = ['Bald','Bangs','Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair','Receding_Hairline','Straight_Hair',\
                    'Wavy_Hair','Wearing_Hat']
        temp_5.append(subnet_2)
        subnet_3 = ['Arched_Eyebrows','Bushy_Eyebrows','Bags_Under_Eyes','Eyeglasses','Narrow_Eyes']
        temp_5.append(subnet_3)
        subnet_4 = ['Big_Nose','Pointy_Nose']
        temp_5.append(subnet_4)
        subnet_5 = ['High_Cheekbones','Rosy_Cheeks','Sideburns','Wearing_Earrings']
        temp_5.append(subnet_5)
        subnet_6 = ['5_o_Clock_Shadow','Big_Lips','Mouth_Slightly_Open','Mustache','Wearing_Lipstick']
        temp_5.append(subnet_6)
        subnet_7 = ['Double_Chin','Goatee','No_Beard']
        temp_5.append(subnet_7)
        subnet_8 = ['Wearing_Necklace','Wearing_Necktie']#,'']
        temp_5.append(subnet_8)
        #classes = [9, 10, 5, 2, 4, 5, 3, 2] 

        temp =list(np.sort(os.listdir(self.arg.path)))
        #print(temp)


        indices = [i for i in range(len(temp))]

        img = []
        count = 0
        for i in temp :
            im = os.path.join(self.arg.path,i)
            #im = imread(os.path.join(self.arg.path,i))
            #im = imresize(imread(os.path.join(self.arg.path,i)),size=(64,64))
            #im = imresize(imread(os.path.join(self.arg.path,i)),size=(256,256))
            #im = (im/127.5)-1
            img.append(im)
            count +=1

        t_img ,te_img ,train_indices ,test_indices = train_test_split(img, indices, test_size=0.09, random_state=15, shuffle=False)

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
        #for i in  data: 
        temp = self.plot(data)
        temp = (temp/127.5)-1
        #a.append(temp)
        return temp

    def bo_batch_norm(self,x, is_training, momentum=0.9, epsilon=0.0001):
        x = tf.layers.batch_normalization(x, momentum=momentum, epsilon=epsilon,training=is_training)
        return x

    def off_shelf_cnn(self,x,reuse=False):
        with tf.variable_scope('compressor') as scope:
            if reuse:
                scope.reuse_variables()
            conv1 = ly.conv2d(x,64,kernel_size=7,activation_fn=tf.nn.leaky_relu,stride=2,padding='SAME')#,normalizer_fn=ly.batch_norm)
            pool1 = ly.max_pool2d(conv1,kernel_size=3,stride=2,padding='SAME')
            #bnorm = ly.batch_norm(pool1)
            bnorm = self.bo_batch_norm(pool1,self.is_train)
            conv2a = ly.conv2d(bnorm,64,kernel_size=1,activation_fn=tf.nn.leaky_relu,stride=1,padding='SAME')#,normalizer_fn=ly.batch_norm)
            conv2 = ly.conv2d(conv2a,192,kernel_size=3,activation_fn=tf.nn.leaky_relu,stride=1,padding='SAME')#,normalizer_fn=ly.batch_norm)
            pool2 = ly.max_pool2d(conv2,kernel_size=3,stride=2,padding='SAME')
            #bnorm2 = ly.batch_norm(pool2)
            bnorm2 = self.bo_batch_norm(pool2,self.is_train)
            conv3a = ly.conv2d(bnorm2,192,kernel_size=1,stride=1,padding='SAME',activation_fn=tf.nn.leaky_relu)#,normalizer_fn=ly.batch_norm)
            conv3 = ly.conv2d(conv3a,384,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.leaky_relu)#,normalizer_fn=ly.batch_norm)
            pool3 = ly.max_pool2d(conv3,kernel_size=3,stride=2,padding='SAME')
            conv4a = ly.conv2d(pool3,384,kernel_size=1,stride=1,padding='SAME',activation_fn=tf.nn.leaky_relu)#,normalizer_fn=ly.batch_norm)
            conv4 = ly.conv2d(conv4a,256,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.leaky_relu)#,normalizer_fn=ly.batch_norm)
            conv5a = ly.conv2d(conv4,256,kernel_size=1,stride=1,padding='SAME',activation_fn=tf.nn.leaky_relu)#,normalizer_fn=ly.batch_norm)
            conv5 = ly.conv2d(conv5a,256,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.leaky_relu)#,normalizer_fn=ly.batch_norm)
            conv6a = ly.conv2d(conv5,256,kernel_size=1,stride=1,padding='SAME',activation_fn=tf.nn.leaky_relu)#,normalizer_fn=ly.batch_norm)
            conv6 = ly.conv2d(conv6a,256,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.leaky_relu)
            pool4 = ly.max_pool2d(conv6,kernel_size=3,stride=2,padding='SAME')
            pool4 = ly.flatten(conv5)
        return pool4 #, conv6


    def shallow_generator_conv(self,image,reuse=False):
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
            conv1 = ly.conv2d(image,96,kernel_size=11,stride=4,padding='SAME',activation_fn=tf.nn.relu,weights_initializer=tf.random_normal_initializer(0, 0.02))
            conv1 = self.bo_batch_norm(conv1, self.is_train)
            conv1 = ly.max_pool2d(conv1,kernel_size=3,stride=2,padding='SAME')

            conv2 = ly.conv2d(conv1,256,kernel_size=5,stride=1,padding='SAME',activation_fn=tf.nn.relu,weights_initializer=tf.random_normal_initializer(0, 0.02))
            conv2 = self.bo_batch_norm(conv2, self.is_train)
            conv2 = ly.max_pool2d(conv2,kernel_size=3,stride=2,padding='SAME')

            conv3 = ly.conv2d(conv2,384,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.relu,weights_initializer=tf.random_normal_initializer(0, 0.02))
            conv3 = self.bo_batch_norm(conv3, self.is_train)
            conv3 = ly.max_pool2d(conv3,kernel_size=3,stride=2,padding='SAME')

            conv4 = ly.conv2d(conv3,384,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.relu,weights_initializer=tf.random_normal_initializer(0, 0.02))
            conv4 = self.bo_batch_norm(conv4, self.is_train)
            conv4 = ly.max_pool2d(conv4,kernel_size=3,stride=2,padding='SAME')

            conv5 = ly.conv2d(conv4,256,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.relu,weights_initializer=tf.random_normal_initializer(0, 0.02))
            conv5 = self.bo_batch_norm(conv5, self.is_train)
            conv5 = ly.max_pool2d(conv5,kernel_size=3,stride=2,padding='SAME')

            flat = ly.flatten(conv5)
            fc1 = ly.fully_connected(flat, 4096, activation_fn=tf.nn.relu)
            fc2 = ly.fully_connected(fc1, 4096, activation_fn=tf.nn.relu)
        return fc2 


    '''
    def generator_conv(self,image,reuse=False):
        dim = 32
        with tf.variable_scope('compressor') as scope:
            if reuse : 
                scope.reuse_variables()
            conv1 = ly.conv2d(image,dim*1,kernel_size=3,stride=2,padding='SAME',activation_fn=tf.nn.leaky_relu,weights_initializer=tf.random_normal_initializer(0, 0.02),normalizer_fn=ly.batch_norm)
            conv2 = ly.conv2d(conv1,dim*2,kernel_size=3,stride=2,padding='SAME',activation_fn=tf.nn.leaky_relu,weights_initializer=tf.random_normal_initializer(0, 0.02),normalizer_fn=ly.batch_norm)
            conv3 = ly.conv2d(conv2,dim*4,kernel_size=3,stride=2,padding='SAME',activation_fn=tf.nn.leaky_relu,weights_initializer=tf.random_normal_initializer(0, 0.02),normalizer_fn=ly.batch_norm)
            conv4 = ly.conv2d(conv3,dim*8,kernel_size=3,stride=2,padding='SAME',activation_fn=tf.nn.leaky_relu,weights_initializer=tf.random_normal_initializer(0, 0.02),normalizer_fn=ly.batch_norm)
            latent = ly.fully_connected(tf.reshape(conv4,shape=[-1,4*4*dim*8]),2,activation_fn=tf.nn.leaky_relu)
            #print(conv4)
        return latent 
    '''
    def decoder_conv(self,latent,reuse=False):
        with tf.variable_scope('reconstructor') as scope:
            if reuse:
                scope.reuse_variables()
            #latent = ly.fully_connected(latent,8*8*256,activation_fn =tf.nn.leaky_relu,normalizer_fn=ly.batch_norm,weights_initializer=tf.random_normal_initializer(0, 0.02))
            #latent = tf.reshape(latent,shape=[-1,8,8,256])
            latent = ly.fully_connected(latent,8*8*256, activation_fn =tf.nn.leaky_relu, weights_initializer=tf.random_normal_initializer(0, 0.02))#,normalizer_fn=ly.batch_norm)
            latent = self.bo_batch_norm(latent, self.is_train)
            latent = tf.reshape(latent,shape=[-1,8,8,256])
            dim = 32
            latent = ly.conv2d_transpose(latent,dim*4,kernel_size=3,stride=2,padding='SAME',activation_fn=tf.nn.leaky_relu)#,normalizer_fn=ly.batch_norm,weights_initializer=tf.random_normal_initializer(0, 0.02))
            unsample1 = ly.conv2d_transpose(latent,dim*4,kernel_size=3,stride=2,padding='SAME',activation_fn=tf.nn.leaky_relu)#,normalizer_fn=ly.batch_norm,weights_initializer=tf.random_normal_initializer(0, 0.02))
            upsample2 = ly.conv2d_transpose(unsample1,dim*2,kernel_size=3,stride=2,padding='SAME',activation_fn=tf.nn.leaky_relu)#,normalizer_fn=ly.batch_norm,weights_initializer=tf.random_normal_initializer(0, 0.02))
            upsample5 = ly.conv2d_transpose(upsample2 ,dim*1,kernel_size=3,stride=2,padding='SAME',activation_fn=tf.nn.leaky_relu)#,normalizer_fn=ly.batch_norm,weights_initializer=tf.random_normal_initializer(0, 0.02))
            upsample6 = ly.conv2d_transpose(upsample5 ,3,kernel_size=3,stride=2,padding='SAME',activation_fn=tf.nn.tanh,weights_initializer=tf.random_normal_initializer(0, 0.02))
        return upsample6  

    def utility_classifier(self,latent,reuse=False):

        temp_5 = []
        subnet_1 = ['Attractive', 'Blurry', 'Chubby', 'Heavy_Makeup', 'Male', 'Oval_Face', 'Pale_Skin','Smiling','Young']
        temp_5.append(subnet_1)
        subnet_2 = ['Bald','Bangs','Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair','Receding_Hairline','Straight_Hair',\
                    'Wavy_Hair','Wearing_Hat']
        temp_5.append(subnet_2)
        subnet_3 = ['Arched_Eyebrows','Bushy_Eyebrows','Bags_Under_Eyes','Eyeglasses','Narrow_Eyes']
        temp_5.append(subnet_3)
        subnet_4 = ['Big_Nose','Pointy_Nose']
        temp_5.append(subnet_4)
        subnet_5 = ['High_Cheekbones','Rosy_Cheeks','Sideburns','Wearing_Earrings']
        temp_5.append(subnet_5)
        subnet_6 = ['5_o_Clock_Shadow','Big_Lips','Mouth_Slightly_Open','Mustache','Wearing_Lipstick']
        temp_5.append(subnet_6)
        subnet_7 = ['Double_Chin','Goatee','No_Beard']
        temp_5.append(subnet_7)
        subnet_8 = ['Wearing_Necklace','Wearing_Necktie']#,'']
        temp_5.append(subnet_8)

        qq = dict()
        for i in range(8):
            qq['subnet_'+str(i)] = temp_5[i]

        output = []
        prob = []
        classes = [9, 10, 5, 2, 4, 5, 3, 2] 
        with tf.variable_scope('utility_classifier') as scope:
            if reuse:
                scope.reuse_variables()
            for i in range(8):
                with tf.variable_scope('subnet_'+str(i+1)):  
                    classifier = ly.fully_connected(latent, 64, activation_fn=tf.nn.relu,weights_initializer=tf.random_normal_initializer(0, 0.02))
                    classifier = self.bo_batch_norm(classifier, self.is_train)
                    classifier = tf.nn.dropout(classifier,keep_prob = self.keep_prob)
                    classifier = ly.fully_connected(classifier, 64, activation_fn=tf.nn.relu)
                    #classifier = self.bo_batch_norm(classifier, self.is_train)
                    #classifier = tf.nn.dropout(classifier, keep_prob = self.keep_prob)
                    for j in qq['subnet_'+str(i)]:
                        with tf.variable_scope(j):
                            classifier = ly.fully_connected(classifier, 2, activation_fn=None)
                            softmax = tf.nn.softmax(classifier)
                            #classifier = ly.fully_connected(classifier, 1, activation_fn=None)
                            #softmax = tf.nn.sigmoid(classifier)
                            output.append(classifier)
                            prob.append(softmax)
        return output , prob

    def build_model(self):

        tf.reset_default_graph()

        #self.image_p = tf.placeholder(tf.float32,shape=(None,256,256,3))
        self.learning_rate_p = tf.placeholder(tf.float32)
        self.image_p = tf.placeholder(tf.float32,shape=(None, 256, 256, 3))
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

        self.noise_p = tf.placeholder(tf.float32,shape=(None, 256, 256, 3))
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_train = tf.placeholder(tf.bool)
        #self.noisy_img = tf.add(self.image_p,self.noise_p)
        #self.latent = self.generator_conv(self.noisy_img)
        self.latent = self.generator_conv(self.image_p)
        #self.latent = self.shallow_generator_conv(self.image_p)
        #self.latent = self.off_shelf_cnn(self.image_p)
        #self.up = self.decoder_conv(self.latent_no,reuse=False)
        #self.up = self.decoder_conv(self.latent)
        self.logit_list , self.prob_list = self.utility_classifier(self.latent,reuse=False)
        #self.prob = tf.nn.softmax(self.classifier)    
        #self.one_hot = tf.one_hot(self.label_p,2)

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

        x =  x / np.max(x)

        x *= 255  

        x= x.astype(np.uint8)

        x = x.reshape(256, 256, 3)

        return x 

    def plot_10slot(self,name):

        r ,c  = 2,10

        j = np.array(self.read(self.te_data[:128]))
        k = np.array([i for i in range(128)])
        penal = np.array([[0.5,1] for i in range(128)])
        no = np.random.normal(size=(128,256,256,3))
        uu = self.sess.run(self.latent, feed_dict={self.image_p:j.reshape(128, 256, 256, 3),self.keep_prob:1,self.is_train:False})
        yy = self.sess.run(self.up, feed_dict={self.latent:uu, self.is_train:False})

        plt.figure(figsize=(10, 2))

        n = 10

        for i in range(n):

            # display original

            ax = plt.subplot(2, n, i + 1)

            #plt.imshow(self.plot(self.te_data[i]))

            plt.imshow(self.plot(j[i]))

            ax.get_xaxis().set_visible(False)

            ax.get_yaxis().set_visible(False)

            # display reconstruction

            ax = plt.subplot(2, n, i + 1 + n)

            plt.imshow(self.plot(yy[i]))

            ax.get_xaxis().set_visible(False)

            ax.get_yaxis().set_visible(False)

        plt.savefig('multcpagn_log/reconstructed_image'+str(name))
        plt.close()
        #plt.show()

    def train(self):
        num_epochs = 80
        train_size = len(self.t_data)
        num_steps_per_epoch = train_size // 256
        num_steps = num_epochs * num_steps_per_epoch
        print(num_steps)
        self.curr_learning_rate = self.init_learning_rate

        acc_trace = [] 
        mse_trace = [] 
        epochs = 80
        #os.mkdir('Male_2_test')
        #f = open('Male_2_test/Male_noise_log.txt','w')
        loss_trace = []
        self.sess.run(self.assign_op)

        count = 0

        for i in range(epochs):

            citers = 1

            epoch_loss = []

            start = time.time()

            for j , k in self.train_next_batch(self.t_data,self.t_label):
                #self.plot_10slot(i)
                b = j.shape[0]
                no = np.random.normal(size=(b, 256, 256,3))

                feed_dict = {}
                feed_dict[self.image_p] = j

                for attr in range(40):
                    feed_dict[self.label_list[attr]] = np.array(k[attr]).reshape(-1)

                feed_dict[self.keep_prob] = 1
                feed_dict[self.is_train] = True
                feed_dict[self.noise_p] = no
                feed_dict[self.learning_rate_p] = self.curr_learning_rate
                #for _ in range(citers):
                #    d_loss , _= self.sess.run([self.loss_r, self.r_opt], feed_dict = feed_dict)

                c_loss , _= self.sess.run([self.loss_c, self.c_opt], feed_dict = feed_dict)

                #for _ in range(3):

                #_, lo = self.sess.run([self.g_opt, self.loss_r], feed_dict = feed_dict)

                count += 1 

            self._update_learning_rate_cosine(count,num_steps)
            end = time.time()

            print('Batch training loss is {}.'.format(c_loss))
            #print('{}/{} epochs , cost {} sec , the utility_loss = {}. reconstruction loss:{}'.format(i+1,epochs,end-start,c_loss,lo))

            #tr_acc = self.compute_acc_test()

            av_acc = self.compute_acc(self.te_data,self.te_label,'test')

            #at_acc = self.compute_acc(self.t_data,self.t_label,'test')

            print('{}/{} epochs, cost {} sec, testing accuracy: {}'.format(i+1, epochs,end-start, av_acc))

            print('{} is the average accuracy of the 40 attributes (testing).'.format(np.mean(av_acc)))

            if (i+1) % 5 == 0 :
                self.saver.save(self.sess,'multcpagn_log/model_multi')
                #self.plot_10slot(i+1)

            #f.write('Average of TPR and TNR is: '+str(tr_acc)+', average testing accuracy is : '+str(av_acc)+'\n')

            #self.saver.save(self.sess,'Male_2_test/model_'+str(i))    

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

    def resize(self,image):
        a = []
        for i in image : 
            a.append(self.preprocess(imresize(i,(256,256))))
        return a 

    def read(self,image):
        a = []
        for i in image : 
            a.append(self.preprocess(imresize(imread(i),size=(256,256))))
        return a 

    def train_next_batch(self, img, label_list, batch_size=256):
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

        for i in range(0,le,256):
            if i ==  (epo *batch_size) : 
                oo = [[] for i in range(40)]
                for k in range(40):
                    oo[k].append(temp_1[k][i:])    
                #yield np.array(temp[i:]), oo
                yield np.array(self.read(temp[i:])),oo
            else : 
                oo = [[] for i in range(40)]
                for k in range(40):
                    oo[k].append(temp_1[k][i:i+256])
                #yield np.array(temp[i:i+256]), oo
                yield np.array(self.read(temp[i:i+256])), oo
    def next_batch(self,encoder_input,label,batch_size=256):
        le = len(encoder_input)
        epo = le//batch_size
        leftover = le - (epo*batch_size)
        for i in range(0,le,256):
            if i ==  (epo *batch_size) : 
                oo = [[] for i in range(40)]
                for k in range(40):
                    oo[k].append(label[k][i:]) 

                #yield np.array(encoder_input[i:]), oo  #np.array(label[i:])
                yield np.array(self.read(encoder_input[i:])), oo
            else : 
                oo = [[] for i in range(40)]
                for k in range(40):
                    oo[k].append(label[k][i:i+256])  
                #yield np.array(encoder_input[i:i+256]), oo  #np.array(label[i:i+256])
                yield np.array(self.read(encoder_input[i:i+256])), oo

    def compute_acc(self,te_data,te_label,name):
        acc_list = []
        pred_list = [[] for i in range(40)]
        for j , k in self.next_batch(te_data,te_label):
            b = j.shape[0]
            no = np.random.normal(size=(b, 256, 256, 3))
            feed_dict = {}
            feed_dict[self.image_p] = j.reshape(b, 256, 256, 3)
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
        #self.saver.restore(self.sess,'log/model_multi')
        self.saver.restore(self.sess,self.arg.model_path)
        self.plot_10slot('test')
            
        av_acc = self.compute_acc(self.te_data,self.te_label,'test')
        print('Testing accuracy is {}.'.format(av_acc))
        a = []
        for i in range(len(self.theta_g)):
            a.append(self.sess.run(self.theta_g[i]))
        np.save('multcpagn_log/theta_g.npy',a)



