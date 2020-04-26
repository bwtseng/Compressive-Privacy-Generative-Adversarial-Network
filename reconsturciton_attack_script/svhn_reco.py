import tensorflow as tf 
import tensorflow.contrib.layers as ly 
import numpy as np
import os 
import time
import random
from keras.datasets import cifar100
import matplotlib.pyplot as plt 
import pandas as pd 
#from tensorflow.python.keras_impl.keras.datasets.cifar10 import load_data
from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data
import math
from scipy.misc import imrotate ,imread ,imsave,imresize
from keras.datasets import cifar10
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from scipy.io import loadmat


plt.switch_backend('agg')
class cpgan:
    def __init__(self,arg):

        self.arg = arg

        self.batch_size = 128

        self.init_learning_rate = 0.01

        self.t_data ,self.t_label ,self.te_data ,self.te_label = self.load_data()

        num_steps_per_epoch = len(self.t_data) // self.batch_size

        #self.num_steps = 1800 * num_steps_per_epoch
        self.num_steps = 10000 * num_steps_per_epoch

        self.build_model()

        #self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob,1),self.label_p),tf.float32))

        #utility_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot,logits=self.logit))

        self.loss_r = tf.losses.mean_squared_error(self.image_p,self.up) 

        #self.loss_c = utility_loss

        #self.loss_g = utility_loss - tf.losses.mean_squared_error(self.image_p,self.up) 

        self.theta_r = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='reconstructor')

        self.theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='compressor')

        uti_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(uti_update):
            self.r_op = tf.train.AdamOptimizer(self.learning_rate_p)
            #self.r_op = tf.train.MomentumOptimizer(self.learning_rate_p,0.9, use_nesterov=True)
            self.r_opt = self.r_op.minimize(self.loss_r, var_list=self.theta_r)

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=15)

        self.assign_op = []
        
        
        a = np.load('svhn_log/theta_g.npy')

        for i in range(len(self.theta_g)):
            if a[i].shape[0] > 3 :
                self.assign_op.append(tf.assign(self.theta_g[i],a[i].reshape(3,3,3,3)))
            else : 
                self.assign_op.append(tf.assign(self.theta_g[i],a[i]))

        print('Length of all parameters in utility_classifier should be assigned:{}'.format(len(self.assign_op)))
        
    def preprocess(self,img):
        svhn_mean = np.array([x / 255.0 for x in[109.9, 109.7, 113.8]])
        svhn_std = np.array([x / 255.0 for x in [50.1, 50.6, 50.8]])
        img = img/127.5
        img = img-1
        #img = img/255
        #img = img - svhn_mean
        #img = img / svhn_std
        return img

    def load_data(self):
        svhn_mean = np.array([x / 255.0 for x in[109.9, 109.7, 113.8]])
        svhn_std = np.array([x / 255.0 for x in [50.1, 50.6, 50.8]])
        '''
        dataset = load_svhn()
        t_data = dataset.train.images
        t_label = dataset.train.labels 
        te_data = dataset.test.images
        te_label = dataset.test.labels
        t_label = np.argmax(t_label,1)
        te_label = np.argmax(te_label,1)
        '''
        temp = loadmat('svhn/train_32x32.mat')
        t_data = temp['X'].astype("float32")
        t_label = temp['y']
        a = []
        b = []
        for i in range(t_data.shape[3]) : 
            #a.append(self.preprocess(i))
            #te = t_data[:, :, :, i]/255.0
            te = self.preprocess(t_data[:,:,:,i])
            a.append(te)
            b.append(t_label[i][0]-1)

        temp = loadmat('svhn/test_32x32.mat')
        te_data = temp['X'].astype("float32")
        te_label = temp['y']
        c = []
        d = []
        for i in range(te_data.shape[3]) : 
            te = self.preprocess(te_data[:,:,:,i])
            #te = te_data[:, :, :, i]/255.0
            c.append(te)
            d.append(te_label[i][0]-1)    

        temp = loadmat('svhn/extra_32x32.mat')
        temp_1 = temp['X'].astype("float32")
        temp_2 = temp['y']
        e = []
        f = []
        for i in range(temp_1.shape[3]):
            te = self.preprocess(temp_1[:,:,:,i])
            #te = temp_1[:, :, :, i]/255.0
            e.append(te)
            f.append(temp_2[i][0]-1)   
            #a.append(self.preprocess(temp_1[i]))
            #f.append(temp_2[i][0])

        a = a + e 
        b = b + f
        return a,b,c,d

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
                left_pad = difference / 2
                right_pad = difference - left_pad
                identity = tf.pad(input_, [[0, 0], [0, 0], [0, 0], [left_pad, right_pad]])
                return x + identity
            else:
                print("Not implemented error")
                exit(1)
        else:
            return x + input_

    def init_tensor(self, shape):
        return tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=1.0))

    def build_model(self):

        self.image_p = tf.placeholder(tf.float32,shape=[None,32,32,3])

        self.label_p = tf.placeholder(tf.int64,shape=[None])

        self.is_train = tf.placeholder(tf.bool)

        self.learning_rate_p = tf.placeholder(tf.float32)
        
        #self.latent = self.generator_conv(self.image_p)
        #self.latent = self.residual_g(self.image_p)
        ### if R is idenetity ?? 
        #self.up = self.decoder_conv(self.latent)

        #self.logit = self.utility_classifier(self.image_p,10)

        self.latent = self.residual_g(self.image_p)

        self.up = self.decoder_conv(self.latent)

        #self.up = self.decoder_conv(self.image_p)

        #self.logit = self.utility_classifier(self.latent,10)
        #self.prob = tf.nn.softmax(self.logit)
        #self.one_hot = tf.one_hot(self.label_p,10)

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
                                x = self._residual_unit(x, 3, filter_size[i], stride[i])
                            else:
                                x = self._residual_unit(x, filter_size[i - 1], filter_size[i], stride[i])
                        else:
                            x = self._residual_unit(x, filter_size[i], filter_size[i], stride[i])
            print(x)
            return x 

    def generator_conv(self,image,reuse=False):
        dim = 32
        with tf.variable_scope('compressor') as scope:
            if reuse : 
                scope.reuse_variables()
            conv1 = ly.conv2d(image,dim*1,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.leaky_relu,weights_initializer=tf.random_normal_initializer(0, 0.02))#,normalizer_fn=ly.batch_norm)
            conv1 = self.bo_batch_norm(conv1,self.is_train)
            conv2 = ly.conv2d(conv1,dim*2,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.leaky_relu,weights_initializer=tf.random_normal_initializer(0, 0.02))#,normalizer_fn=ly.batch_norm)
            conv2 = self.bo_batch_norm(conv2,self.is_train)
            conv3 = ly.conv2d(conv2,dim*4,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.leaky_relu,weights_initializer=tf.random_normal_initializer(0, 0.02))#,normalizer_fn=ly.batch_norm)
            conv4 = self.bo_batch_norm(conv3,self.is_train)
            conv4 = ly.conv2d(conv3,dim*8,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.leaky_relu,weights_initializer=tf.random_normal_initializer(0, 0.02))#,normalizer_fn=ly.batch_norm)
            conv4 = self.bo_batch_norm(conv4,self.is_train)
            latent = ly.conv2d(conv4,3,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.leaky_relu,weights_initializer=tf.random_normal_initializer(0, 0.02))#,normalizer_fn=ly.batch_norm)
            #latent = ly.fully_connected(tf.reshape(conv4,shape=[-1,7*7*dim*8]),2,activation_fn=tf.nn.leaky_relu)
            print(latent)
        return latent 
        
    def decoder_conv(self,latent,reuse=False):
        with tf.variable_scope('reconstructor') as scope:
            if reuse:
                scope.reuse_variables()
            dim = 32
            latent = ly.flatten(latent)
            latent = ly.fully_connected(latent, 4*4*64, activation_fn=tf.nn.relu)
            latent = self.bo_batch_norm(latent, self.is_train)
            latent = tf.reshape(latent, shape=[-1,4,4,64])
            #latent = tf.reshape(latent, shape=[-1, 4, 4, 32])
            upsample1 = ly.conv2d_transpose(latent, dim*4, kernel_size=3, stride=2, padding='SAME',activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
            upsample1 = self.bo_batch_norm(upsample1, self.is_train)
            upsample2 = ly.conv2d_transpose(upsample1, dim*2, kernel_size=3, stride=2, padding='SAME',activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
            upsample2 = self.bo_batch_norm(upsample2, self.is_train)
            #upsample5 = ly.conv2d_transpose(upsample2, dim*1, kernel_size=3, stride=1, padding='SAME',activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
            upsample6 = ly.conv2d_transpose(upsample2, 3, kernel_size=3, stride=2, padding='SAME',activation_fn=tf.nn.tanh, weights_initializer=tf.random_normal_initializer(0, 0.02))
        return upsample6 
    

    def wideres33block(self,X,N,K,iw,bw,s,scope):
        
        # Creates N no. of 3,3 type residual blocks with dropout that consitute the conv2/3/4 blocks
        # with widening factor K and X as input. s is stride and bw is base width (no. of filters before multiplying with k)
        # iw is input width.
        # (see https://arxiv.org/abs/1605.07146 paper for details on the block)
        # In this case, dropout = probability to keep the neuron enabled.
        # phase = true when training, false otherwise.
        
        with tf.variable_scope(scope):
            conv33_1 = ly.conv2d(X,kernel_size=3,num_outputs=bw*K,padding='SAME',stride=s,activation_fn=None)
            conv33_1 = self.bo_batch_norm(conv33_1,self.is_train)
            conv33_1 = tf.nn.dropout(conv33_1,keep_prob = self.dropout)
            
            conv33_2 = ly.conv2d(conv33_1,kernel_size=3,num_outputs=bw*K,stride=1,activation_fn=None)
            conv_skip= ly.conv2d(X,kernel_size=1,num_outputs=bw*K,stride=s,activation_fn=None) #shortcut connection

            caddtable = tf.add(conv33_2,conv_skip)
            
            #1st of the N blocks for conv2/3/4 block ends here. The rest of N-1 blocks will be implemented next with a loop.

            for i in range(0,N-1):
                
                C = caddtable
                Cactivated = self.bo_batch_norm(C,self.is_train)
                
                conv33_1 = ly.conv2d(Cactivated,kernel_size=3,num_outputs=bw*K,stride=1,activation_fn=None)
                conv33_1 = self.bo_batch_norm(conv33_1,self.is_train)
                conv33_1 = tf.nn.dropout(conv33_1,self.dropout)
                conv33_2 = ly.conv2d(conv33_1,kernel_size=3,num_outputs=bw*K,stride=1,activation_fn=None)
                caddtable = tf.add(conv33_2,C)
            out = self.bo_batch_norm(caddtable,self.is_train)
        return out 

    def WRN(self,x,layers,K):#,scope): #Wide residual network

        # 1 conv + 3 convblocks*(3 conv layers *1 group for each block + 2 conv layers*(N-1) groups for each block [total 1+N-1 = N groups]) = layers
        # 3*2*(N-1) = layers - 1 - 3*3
        # N = (layers -10)/6 + 1
        # So N = (layers-4)/6

        N = int((layers-4)/6)
        conv1 = ly.conv2d(x,kernel_size=3,num_outputs=16,stride=1,activation_fn=None)
        conv1 = self.bo_batch_norm(conv1,self.is_train)
        conv2 = self.wideres33block(conv1,N,K,16,16,1,'conv2')
        conv3 = self.wideres33block(conv2,N,K,16*K,32,2,'conv3')
        conv4 = self.wideres33block(conv3,N,K,32*K,64,2,'conv4')

        pooled = tf.nn.avg_pool(conv4,ksize=[1,8,8,1],strides=[1,1,1,1],padding='VALID')
        
        #Initialize weights and biases for fully connected layers
        #with tf.variable_scope(scope+"regularize",reuse=False):
        #   wd1 = tf.Variable(tf.truncated_normal([1*1*64*K,64*K],stddev=5e-2))
        #    wout = tf.Variable(tf.truncated_normal([64*K, n_classes]))
        #bd1 = tf.Variable(tf.constant(0.1,shape=[64*K]))
        #bout = tf.Variable(tf.constant(0.1,shape=[n_classes]))

        # Fully connected layer
        # Reshape pooling layer output to fit fully connected layer input
        wd1 = tf.Variable(tf.truncated_normal([1*1*64*K,64*K],stddev=5e-2))
        #sha = pooled.get_shape().as_list()
        #sha = (sha[1],sha[2],sha[3])
        fc1 = tf.reshape(pooled, [-1, wd1.get_shape().as_list()[0]])   
        #fc1 = tf.reshape(fc1,sha)
        #fc1 = ly.flatten(pooled)
        with tf.variable_scope('logits'):
            fc1 = ly.fully_connected(fc1,64*K,activation_fn=tf.nn.relu)
            #fc1 = tf.add(tf.matmul(fc1, wd1), bd1)
            #fc1 = tf.nn.elu(fc1)
            # Output, class prediction
            #out = tf.add(tf.matmul(fc1, wout), bout)
            out = ly.fully_connected(fc1,10,activation_fn=None)
        return out


    def wrs_16_2(self,img):
        #widing_factor = 2
        widing_factor = 8
        num_residual_units = 2 # N = (deep -4 /6)
        print('Building model')
        # Init. conv.
        x = ly.conv2d(img,kernel_size=3,num_outputs=16, stride=1,activation_fn=None)
        # Residual Blocks
        filters = [16, 16*widing_factor, 32*widing_factor, 64*widing_factor]
        strides = [1, 2, 2]


        with tf.variable_scope('utility_classifier'):
            for i in range(1, 4):
                # First residual unit
                with tf.variable_scope('unit_%d_0' % i) as scope:
                    print('\tBuilding residual unit: %s' % scope.name)
                    x = self.bo_batch_norm(x,self.is_train)

                    # Shortcut
                    if filters[i-1] == filters[i]:
                        if strides[i-1] == 1:
                            shortcut = tf.identity(x)
                        else:
                            shortcut = tf.nn.max_pool(x, [1, strides[i-1], strides[i-1], 1],
                                                      [1, strides[i-1], strides[i-1], 1], 'VALID')
                    else:
                        #shortcut = utils._conv(x, 1, filters[i], strides[i-1], name='shortcut')
                        shortcut = ly.conv2d(x,kernel_size=1, num_outputs=filters[i], stride=strides[i-1])
                    # Residual
                    x = ly.conv2d(x, kernel_size=3, num_outputs=filters[i], stride=strides[i-1],activation_fn=None)
                    x = self.bo_batch_norm(x, self.is_train)
                    x = tf.nn.dropout(x,keep_prob=self.dropout)
                    x = ly.conv2d(x, kernel_size=3, num_outputs=filters[i], stride=1,activation_fn=None)

                    # Merge
                    x = x + shortcut
                # Other residual units
                for j in range(1,num_residual_units):
                    with tf.variable_scope('unit_%d_%d' % (i, j)) as scope:
                        print('\tBuilding residual unit: %s' % scope.name)
                        # Shortcut
                        shortcut = x
                        # Residual
                        x = self.bo_batch_norm(x,self.is_train)
                        x = ly.conv2d(x, kernel_size=3, num_outputs=filters[i], stride=1,activation_fn=None)
                        x = self.bo_batch_norm(x,self.is_train)
                        x = tf.nn.dropout(x,keep_prob=self.dropout)
                        x = ly.conv2d(x, kernel_size=3, num_outputs=filters[i], stride=1,activation_fn=None)
                        # Merge
                        x = x + shortcut


            # Last unit
            with tf.variable_scope('unit_last') as scope:
                print('\tBuilding unit: %s' % scope.name)
                x = self.bo_batch_norm(x, self.is_train)
                x = tf.nn.avg_pool(x,ksize=[1,8,8,1],strides=[1,1,1,1],padding='VALID')
                #sha = x.get_shape().as_list()
                #fc1 = tf.reshape(x, [-1, sha[1]])  
                #x = tf.reduce_mean(x, [1, 2])

            # Logit
            with tf.variable_scope('logits') as scope:
                print('\tBuilding unit: %s' % scope.name)
                x_shape = x.get_shape().as_list()
                #x = tf.reshape(x, [-1, x_shape[1]])
                print(x_shape)
                x = tf.reshape(x, [-1, x_shape[-1]])
                x = ly.fully_connected(x,10,activation_fn=None)
        return x

    def bo_batch_norm(self,x, is_training, momentum=0.9, epsilon=0.00001):
        """
        Add a new batch-normalization layer.
        :param x: tf.Tensor, shape: (N, H, W, C).
        :param is_training: bool, train mode : True, test mode : False
        :return: tf.Tensor.
        """
        x = tf.layers.batch_normalization(x, momentum=momentum, epsilon=epsilon,training=is_training)
        return x

    def init_tensor(self, shape):
        return tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=1.0))

    def _conv(self, input, filter_shape, stride):
        """Convolutional layer"""
        return tf.nn.conv2d(input,
                            filter=self.init_tensor(filter_shape),
                            strides=[1, stride, stride, 1],
                            padding="SAME")


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
            a[i] -= cifar_mean
            a[i] /= cifar_std

        return a

    def batch_mirror_image(self,image):
        a = []
        for i in image :
            a.append(np.flipud(i))
        return a

    def batch_crop_image(self,image):
        a = [] 
        ind = [1,2]
        '''
        aug = random.sample(ind,1)
        if aug ==2 : 
            image = self.batch_mirror_image(image)
        '''
        for i in image : 
            reflection = bool(np.random.randint(2))
            if reflection:
                 image = np.fliplr(i)
            #else : 
        #    image = image 
            image_pad = np.pad(i,((4,4),(4,4),(0,0)),mode='constant')

            crop_x1 = random.randint(0,8)
            crop_x2 = crop_x1 + 32

            crop_y1 = random.randint(0,8)
            crop_y2 = crop_y1 + 32

            image_crop = image_pad[crop_x1:crop_x2,crop_y1:crop_y2]
            a.append(image_crop)

        return a


    def next_batch(self,t_data,t_label,batch_size=128):
        le = len(t_data)
        epo = le//batch_size
        leftover = le - epo * batch_size
        sup = batch_size - leftover
        for i in range(0,le,128):
            if i ==  (epo *batch_size) : 
                #yield np.array(encoder_input[i:]) , np.array(label[i:])
                yield np.concatenate((t_data[i:],t_data[:sup]),axis=0) , np.concatenate((t_label[i:],t_label[:sup]),axis=0) 
            else : 
                yield np.array(t_data[i:i+128]) , np.array(t_label[i:i+128])

    def t_next_batch(self,t_data,t_label,batch_size=128):
        le = len(t_data)
        epo = le//batch_size
        leftover = le - epo * batch_size
        sup = batch_size - leftover
        #c = list(zip(t_data,t_label))
        #random.shuffle(c)
        #t_data , t_label = zip(*c)
        for i in range(0,le,128):
            #c = [0,1,2,3]
            c = [2,3]
            aug = random.sample(c,1)[0]
            if i ==  (epo *batch_size) : 
                if aug==1 :
                    #yield np.array(t_data[i:]) , np.array(t_label[i:]), np.array(self.batch_random_rotate_image(t_data[i:])) , np.array(t_label[i:]) ,aug
                    yield np.concatenate((t_data[i:],t_data[:sup]),axis=0) , np.concatenate((t_label[i:],t_label[:sup]),axis=0) , np.array(self.batch_random_rotate_image(t_data[i:])+self.batch_mirror_image(t_data[:sup])),\
                            np.concatenate((t_label[i:],t_label[:sup]),axis=0) , aug
                elif aug == 0:
                    #yield np.array(t_data[i:]) , np.array(t_label[i:]), np.array(self.batch_mirror_image(t_data[i:])) , np.array(t_label[i:]) , aug
                    yield np.concatenate((t_data[i:],t_data[:sup]),axis=0) , np.concatenate((t_label[i:],t_label[:sup]),axis=0) , np.array(self.batch_mirror_image(t_data[i:])+self.batch_mirror_image(t_data[:sup])),\
                            np.concatenate((t_label[i:],t_label[:sup]),axis=0) , aug
                elif aug == 2:
                    #yield np.array(t_data[i:]) , np.array(t_label[i:]), np.array(self.batch_crop_image(t_data[i:])) , np.array(t_label[i:]) , aug   
                    yield np.concatenate((t_data[i:],t_data[:sup]),axis=0) , np.concatenate((t_label[i:],t_label[:sup]),axis=0) , np.array(self.batch_crop_image(t_data[i:])+self.batch_mirror_image(t_data[:sup])),\
                            np.concatenate((t_label[i:],t_label[:sup]),axis=0) , aug                
                else :
                    yield np.concatenate((t_data[i:],t_data[:sup]),axis=0) , np.concatenate((t_label[i:],t_label[:sup]),axis=0) , np.concatenate((t_data[i:],t_data[:sup]),axis=0),\
                            np.concatenate((t_label[i:],t_label[:sup]),axis=0) , aug   
            else : 
                if aug == 1:
                    yield np.array(t_data[i:i+128]) , np.array(t_label[i:i+128]), np.array(self.batch_random_rotate_image(t_data[i:i+128])) , np.array(t_label[i:i+128]) ,aug
                elif aug == 0:
                    yield np.array(t_data[i:i+128]) , np.array(t_label[i:i+128]), np.array(self.batch_mirror_image(t_data[i:i+128])) , np.array(t_label[i:i+128]) , aug
                elif aug == 2 : 
                    yield np.array(t_data[i:i+128]) , np.array(t_label[i:i+128]), np.array(self.batch_crop_image(t_data[i:i+128])) , np.array(t_label[i:i+128]) , aug
                else :
                    yield np.array(t_data[i:i+128]) , np.array(t_label[i:i+128]), np.array(t_data[i:i+128]) , np.array(t_label[i:i+128]) ,aug

    def plot_10slot(self,name):

        j = np.array(self.te_data[:128])

        k = np.array([i for i in range(128)])

        no = np.random.normal(size=(128,32,32,3))

        #self.sess.run(self.assign_op)

        #uu = self.sess.run(self.latent, feed_dict={self.image_p:j.reshape(128,32,32,3),self.label_p:k,self.is_train:False})

        #yy = self.sess.run(self.up, feed_dict={self.latent:uu,self.is_train:False})

        yy = self.sess.run(self.up, feed_dict={self.image_p:j.reshape(128,32,32,3), self.is_train:False})


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
           
        plt.savefig('svhn_log/reconstructed_image'+str(name))

    def plot(self,x):

        x = x - np.min(x)

        #x /= np.max(x)
        x = x / np.max(x)

        x *= 255  

        x= x.astype(np.uint8)

        x = x.reshape(32,32,3)

        return x 

    def compute_acc(self,te_data,te_label,is_train=False):

        acc_list = []

        for j , k in self.next_batch(te_data,te_label):

            b = k.shape[0]

            no = np.random.normal(size=(b,32,32,3))

            pred = self.sess.run(self.prob,feed_dict={self.image_p:j.reshape(b,32,32,3),self.label_p:k,self.is_train:False})

            #aaa = self.sess.run(self.acc,feed_dict={self.image_p:j.reshape(b,32,32,3),self.label_p:k,self.is_train:False})

            #acc_list += list(np.argmax(pred,1))
            acc_list.append(pred)

        if is_train :
            preds = np.concatenate((acc_list),axis=0)
            preds = preds[0:50000]          
        else :
            preds = np.concatenate((acc_list),axis=0)
            preds = preds[0:10000]

        ac = accuracy_score(np.argmax(preds,axis=1),te_label)
        #print(ac)
        return ac

    def compute_reco_mse(self):
        error = []
        for i , j in self.next_batch(self.te_data,self.te_label):
            b = j.shape[0]
            no = np.random.normal(size=(b,64,64,3))
            up = self.sess.run(self.up,feed_dict={self.image_p:i.reshape(b,32,32,3),self.is_train:False})
            for k in range(len(up)):
                #error.append(mean_squared_error(self.plot(i[k]).flatten(),self.plot(up[k]).flatten()))
                error.append(mean_squared_error(i[k].flatten(), up[k].flatten()))
        print('Average MSE among all testing images is {}'.format(np.mean(error)))
        return np.mean(error)

    def train(self):
        epochs = 10000
        mse_trace = []
        acc_trace = []
        count = 0
        self.curr_learning_rate = self.init_learning_rate
        for i in range(epochs):
            #citers = 25
            epoch_loss = []
            start = time.time()
            for j , k ,q , l ,aug in self.t_next_batch(self.t_data,self.t_label):
                self.sess.run(self.assign_op)
                if aug != 3 : 
                    b = k.shape[0]
                    no = np.random.normal(size=(b,32,32,3))
                    feed_dict = {self.image_p:j.reshape(b,32,32,3),self.label_p:l,self.is_train:True,self.learning_rate_p:self.curr_learning_rate} 
                else : 
                    b = k.shape[0]
                    no = np.random.normal(size=(b,32,32,3))
                    feed_dict = {self.image_p:j.reshape(b,32,32,3),self.label_p:l,self.is_train:True,self.learning_rate_p:self.curr_learning_rate} 

                #for _ in range(citers):
                d_loss, _ = self.sess.run([self.loss_r, self.r_opt],feed_dict=feed_dict)
                #acc_1,pro,lo , _= self.sess.run([self.acc,self.prob,self.loss_c,self.c_opt],feed_dict=feed_dict)
                count+=1
                #print(acc_1)
                #_, lo_1 = self.sess.run([self.g_opt,self.loss_r],feed_dict=feed_dict)

            end = time.time()

            #self._update_learning_rate_cosine(count,self.num_steps)

            #print('{}/{} epochs , cost {} sec , the utility_loss = {}.,acc = {}'.format(i+1,epochs,end-start,lo,acc_1))

            #at_acc = self.compute_acc(self.t_data,self.t_label,is_train=True)

            #av_acc = self.compute_acc(self.te_data,self.te_label)

            #print('Training accuracy is {}, testing accuracy is {}.lr is {}'.format(at_acc,av_acc,self.curr_learning_rate))

            #acc_trace.append(av_acc)

            #self.saver.save(self.sess,'cifar10/model_'+str(i))    
            a = self.compute_reco_mse()
            mse_trace.append(a)
            if (i+1) % 3 ==0:
                self.plot_10slot(i)    
            np.save('svhn_log/mse_trace.npy',mse_trace)  
        #f.close()

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
        #self.op._lr = new_learning_rate
        self.curr_learning_rate = new_learning_rate

    def test(self):
        temp = np.load('svhn_log/mse_trace.npy')
        le = len(list(temp))
        le = [i for i in range(le)]
        plt.plot(le,temp)
        plt.title('Training curve of the attcker network(SVHN)')
        #plt.yticks([0.5,0.6,0.7,0.8,0.9,1,1.1,1.2])
        plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2])
        #plt.yticks([0.6,0.7,0.8,0.9,1])
        #plt.yticks([0.2,0.22,0.24,0.26,0.28])
        plt.xlabel('Epochs')
        plt.ylabel('Mean square error')
        plt.savefig('svhn_mse_trace.png')


