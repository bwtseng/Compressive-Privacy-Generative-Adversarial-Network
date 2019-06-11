import os
import h5py 
import time
import random
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import tensorflow as tf
import tensorflow.contrib.layers as ly 
from scipy.misc import imread, imresize, imsave
from sklearn.model_selection import train_test_split

plt.switch_backend('agg')
tf.set_random_seed(9)

class cpgan: 
    def __init__(self,args):
        self.arg = args
        self.path = self.arg.path
        self.path_label = self.arg.path_label
        self.attribute = self.arg.attribute 
        self.com_dim = self.arg.com_dim
        all_data, all_label = self.load_data()
        sam = []
        self.t_data ,self.te_data ,self.t_label ,self.te_label = train_test_split(all_data, all_label, test_size=0.05, random_state=9)
        self.build_model()

        ## Model parameter and computation.
        print("The number of multiplication and addtion : {} and {}.".format(self.g_multiplication, self.g_addition))
        print("The number of multiplication and addtion : {} and {}.".format(self.c_multiplication, self.c_addition))

        ## Objective function and evaluation metric.
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob,1), self.label_p), tf.float32))
        utility_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot, logits=self.classifier))
        self.loss_r = tf.losses.mean_squared_error(self.image_p, self.up) 
        self.loss_c = utility_loss
        self.loss_g = utility_loss - tf.losses.mean_squared_error(self.image_p, self.up) 

        ## Get weights.
        self.theta_r = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='reconstructor')
        self.theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='compressor')
        self.theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='utility_classifier')

        ## Optimizer
        self.g_opt = tf.train.AdamOptimizer(0.001).minimize(self.loss_g,var_list=self.theta_g)
        self.c_opt = tf.train.AdamOptimizer(0.001).minimize(self.loss_c,var_list=self.theta_c)
        self.r_opt = tf.train.AdamOptimizer(0.001).minimize(self.loss_r,var_list=self.theta_r)

        print('The numbers of parameters in variable_scope G are : {}'.format(self.count_number_trainable_params(self.theta_g)))
        print('The numbers of parameters in variable_scope C are : {}'.format(self.count_number_trainable_params(self.theta_c)))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=15)

    def load_data(self):
        temp =list(np.sort(os.listdir(self.arg.path)))
        img = []
        count = 0
        for i in temp : 
            img.append(os.path.join(self.arg.path, i))
        lab = []
        temp = pd.read_csv(self.arg.path_label)[['Male']].values.reshape(-1)
        for i in temp : 
            if i == -1:
                lab.append(0)
            else : 
                lab.append(1)
        return img, lab 

    def preprocess(self, path_list):
        img = []
        for path in path_list:
            im = imread(path)
            im = im/127.5 
            im = im -1 
            img.append(im)
        return np.array(img)

    def sample(self, data, all_label):
        ## To avoid the inbalance sample of the class you specify.
        index_1 = []
        index_0 = []
        for i in range(len(all_label)):
            if all_label[i] == 1 : 
                index_1.append(i)
            else : 
                index_0.append(i)
        random.shuffle(index_1)
        random.shuffle(index_0)
        cc = index_1[:(len(index_1)-500)]
        dd = index_1[(len(index_1)-500):]
        ee = index_0[:(len(index_0)-9500)]
        ff = index_0[(len(index_0)-9500):]
        test_amout = 9500 
        t_data = []
        te_data = []
        t_label = []
        te_label = []
        for i in range(202599):
            if i in cc : 
                t_data.append(data[i])
                t_label.append(1)
            elif i in dd : 
                te_data.append(data[i])
                te_label.append(1)
            elif i in ee :
                t_data.append(data[i])
                t_label.append(0)
            else : 
                te_data.append(data[i])
                te_label.append(0)

        return t_data , te_data ,t_label ,te_label

    def bo_batch_norm(self,x, is_training, momentum=0.9, epsilon=0.001):
        x = tf.layers.batch_normalization(x, momentum=momentum, epsilon=epsilon, training=is_training)
        return x

    def generator_conv(self,image,reuse=False):

        dim = 32

        self.g_addition = 0
        self.g_multiplication = 0

        with tf.variable_scope('compressor') as scope:
            if reuse : 
                scope.reuse_variables()

            channel = image.get_shape().as_list()[-1]
            conv1 = ly.conv2d(image,dim*1, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.leaky_relu, weights_initializer=tf.random_normal_initializer(0, 0.02))

            temp = conv1.get_shape().as_list()
            self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
            self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel)

            conv1 = self.bo_batch_norm(conv1, self.is_train)
            temp = conv1.get_shape().as_list()
            self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
            self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel) 


            channel = temp[-1]
            conv2 = ly.conv2d(conv1,dim*2, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.leaky_relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
            temp = conv2.get_shape().as_list()
            self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
            self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel)

            conv2 = self.bo_batch_norm(conv2, self.is_train)

            temp = conv2.get_shape().as_list()
            self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
            self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel) 


            channel = temp[-1]
            conv3 = ly.conv2d(conv2,dim*4, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.leaky_relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
            temp = conv3.get_shape().as_list()
            
            self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
            self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel)

            conv3 = self.bo_batch_norm(conv3, self.is_train)

            temp = conv3.get_shape().as_list()
            self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
            self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel) 


            channel = temp[-1]

            conv4 = ly.conv2d(conv3,dim*8, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.leaky_relu, weights_initializer=tf.random_normal_initializer(0, 0.02))

            self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
            self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel)

            conv4 = self.bo_batch_norm(conv4, self.is_train)

            temp = conv4.get_shape().as_list()
            self.g_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
            self.g_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel) 

            #latent = ly.fully_connected(tf.reshape(conv4,shape=[-1,4*4*dim*8]),4096,activation_fn=tf.nn.leaky_relu)
            x_shape = tf.reshape(conv4, shape=[-1,7*7*dim*8]).get_shape().as_list()

            self.g_addition += (x_shape[-1]*2)
            self.g_multiplication += (x_shape[-1]*2)

            latent = ly.fully_connected(tf.reshape(conv4, shape=[-1,7*7*dim*8]), self.com_dim, activation_fn=tf.nn.leaky_relu)
        return latent 

    def decoder_conv(self,latent,reuse=False):
        with tf.variable_scope('reconstructor') as scope:
            if reuse:
                scope.reuse_variables()

            latent = ly.fully_connected(latent, 7*7*256, activation_fn =tf.nn.leaky_relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
            latent = self.bo_batch_norm(latent, self.is_train)
            latent = tf.reshape(latent,shape=[ -1, 7, 7, 256])
            dim = 32
            unsample1 = ly.conv2d_transpose(latent, dim*4, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.leaky_relu)
            upsample2 = ly.conv2d_transpose(unsample1, dim*2, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.leaky_relu)
            upsample3 = ly.conv2d_transpose(upsample2 ,dim*1, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.leaky_relu)
            upsample4 = ly.conv2d_transpose(upsample3 ,3, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.tanh, weights_initializer=tf.random_normal_initializer(0, 0.02))
        return upsample6 4



    def utility_classifier(self, latent, classes, reuse=False):

        self.c_addition = 0
        self.c_multiplication = 0
        with tf.variable_scope('utility_classifier') as scope:
            if reuse:
                scope.reuse_variables()

            x_shape = latent.get_shape().as_list()
            classifier= ly.fully_connected(latent, 256, activation_fn=tf.nn.leaky_relu, weights_initializer=tf.random_normal_initializer(0, 0.02))

            self.c_addition += (x_shape[-1]*256)
            self.c_multiplication += (x_shape[-1]*256)
            

            classifier = self.bo_batch_norm(classifier, self.is_train)
            temp = classifier.get_shape().as_list()
            self.c_addition += (temp[1])
            self.c_multiplication += (temp[1]) 


            #self.c_addition +=1 
            #self.c_multiplication +=1
            x_shape = classifier.get_shape().as_list()
            classifier = tf.nn.dropout(classifier, keep_prob = self.keep_prob)
            classifier= ly.fully_connected(classifier, 256, activation_fn=tf.nn.leaky_relu, weights_initializer=tf.random_normal_initializer(0, 0.02))

            self.c_addition += (x_shape[-1]*256)
            self.c_multiplication += (x_shape[-1]*256)

            classifier = self.bo_batch_norm(classifier, self.is_train)
            temp = classifier.get_shape().as_list()
            self.c_addition += (temp[1])
            self.c_multiplication += (temp[1]) 
            classifier = tf.nn.dropout(classifier, keep_prob=self.keep_prob)

            x_shape = classifier.get_shape().as_list()
            classifier= ly.fully_connected(classifier, classes, activation_fn=None)

            self.c_addition += (x_shape[-1]*classes)
            self.c_multiplication += (x_shape[-1]*classes)

        return classifier



    def build_model(self):

        tf.reset_default_graph()

        self.image_p = tf.placeholder(tf.float32, shape=(None, 112, 112, 3))
        self.label_p = tf.placeholder(tf.int64, shape=(None)) 
        self.one_hot = tf.one_hot(self.label_p, 2)
        self.noise_p = tf.placeholder(tf.float32, shape=(None, 112, 112, 3))
        self.keep_prob = tf.placeholder(tf.float32)
        self.penalty = tf.placeholder(tf.float32, shape=(None, 2))
        self.is_train = tf.placeholder(tf.bool)
        self.noisy_img = tf.add(self.image_p, self.noise_p)
        self.latent_no = self.generator_conv(self.noisy_img)
        self.up = self.decoder_conv(self.latent_no, reuse=False)
        self.classifier = self.utility_classifier(self.latent_no, 2, reuse=False)
        self.prob = tf.nn.softmax(self.classifier)    
    
        ### similar to weighted softmax and focal loss form.
        #penalty = tf.cast(tf.constant(np.array([5,1]).reshape(2,1)),tf.float32)
        #self.one_hot =  self.penalty * tf.cast(tf.one_hot(self.label_p,2),tf.float32)
        #self.one_hot = tf.cast(self.one_hot,tf.int64)
        

    def compute_acc(self, te_data, te_label):
        acc_list = []
        for j , k in self.next_batch(te_data, te_label):
            b = k.shape[0]
            penal = np.array([[5,1] for i in range(b)])
            no = np.random.normal(size=(b, 112, 112, 3))
            pred = self.sess.run(self.prob, feed_dict={self.image_p:j.reshape(b, 112, 112, 3), self.label_p:k, self.noise_p:no, self.keep_prob:1, self.penalty:penal})
            acc_list += list(np.argmax(pred,1))
        correct = []
        for i in range(len(acc_list)):
            if acc_list[i] == te_label[i]:
                correct.append(1)
        return len(correct)/len(te_label)



    def compute_acc_test(self):
        acc_list = []
        for j , k in self.next_batch(self.te_data,self.te_label):
            b = k.shape[0]
            penal = np.array([[5,1] for i in range(b)])
            no = np.random.normal(size=(b,64,64,3))
            pred = self.sess.run(self.prob,feed_dict={self.image_p:j.reshape(b,112, 112, 3), self.label_p:k, self.noise_p:no, self.keep_prob:1, self.penalty:penal})
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



    def plot(self, x):
        x = x - np.min(x)
        x /= np.max(x)
        x *= 255  
        x= x.astype(np.uint8)
        x = x.reshape(112, 112, 3)
        return x 

    def plot_10slot(self, name):
        r ,c  = 2,10
        j = np.array(self.te_data[:128])
        k = np.array([i for i in range(128)])
        penal = np.array([[0.5,1] for i in range(128)])
        no = np.random.normal(size=(128, 112, 112, 3))
        uu = self.sess.run(self.latent_no, feed_dict={self.image_p:j.reshape(128,64,64,3), self.label_p:k, self.noise_p:no, self.keep_prob:1, self.penalty:penal})
        yy = self.sess.run(self.up, feed_dict={self.latent_no:uu})

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
            plt.imshow(self.plot(yy[i]))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.savefig('single_celebA/reconstructed_image'+str(name))
        plt.close()
        #plt.show()

    def train(self):
        acc_trace = [] 
        mse_trace = [] 
        epochs = 15        
        # os.mkdir('Male_2_test')
        # f = open('single_celebA/Male_log.txt','w')
        for i in range(epochs):
            citers = 25
            epoch_loss = []
            start = time.time()
            for j , k in self.train_next_batch(self.t_data,self.t_label):
                b = k.shape[0]
                no = np.random.normal(size=(b,64,64,3))
                penal = np.array([[5,1] for i in range(b)])
                for _ in range(citers):
                    d_loss, _= self.sess.run([self.loss_r, self.r_opt], feed_dict={self.image_p:j.reshape(b, 112, 112, 3),self.label_p:k, self.noise_p:no, self.keep_prob:0.5, self.penalty:penal})
                acc_1, c_loss, _= self.sess.run([self.acc, self.loss_c, self.c_opt],feed_dict={self.image_p:j.reshape(b, 112, 112, 3), self.label_p:k, self.noise_p:no, self.keep_prob:0.5, self.penalty:penal})
                _, lo = self.sess.run([self.g_opt, self.loss_r],feed_dict={self.image_p:j.reshape(b, 112, 112, 3), self.label_p:k, self.noise_p:no, self.keep_prob:0.5, self.penalty:penal})

            end = time.time()
            print('{}/{} epochs , cost {} sec , the utility_loss = {}.,acc = {}'.format(i+1,epochs, end-start, lo, acc_1))
            tr_acc = self.compute_acc_test()
            av_acc = self.compute_acc(self.te_data,self.te_label)

            #f.write('Average of TPR and TNR is: '+str(tr_acc)+', average testing accuracy is : '+str(av_acc)+'\n')
            self.saver.save(self.sess,'model_'+str(i))    
            self.plot_10slot(i)      

        #f.close()

    def test(self):
        self.saver.restore(self.sess, self.arg.model_path)
        #self.plot_10slot('restore')  
        np.save('single_celebA/g.npy', self.sess.run(self.theta_g))
        self.compute_acc_test()

    def train_next_batch(self, encoder_input, label, batch_size=512):
        le = len(encoder_input)
        epo = le//batch_size
        leftover = le - (epo*batch_size)
        c = list(zip(encoder_input,label))
        random.shuffle(c)
        temp , temp_1 = zip(*c)
        for i in range(0,le,512):
            if i ==  (epo *batch_size) : 
                yield self.preprocess(temp[i:]) , np.array(temp_1[i:])
            else : 
                yield self.preprocess(temp[i:i+512]) , np.array(temp_1[i:i+512])


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



    def next_batch(self,encoder_input,label,batch_size=256):
        le = len(encoder_input)
        epo = le//batch_size
        leftover = le - (epo*batch_size)
        for i in range(0,le,512):
            if i ==  (epo *batch_size) : 
                #yield np.array(encoder_input[i:]) , np.array(label[i:])
                yield self.preprocess(encoder_input[i:]) , np.array(label[i:])
            else : 
                #yield np.array(encoder_input[i:i+256]) , np.array(label[i:i+256])
                yield self.preprocess(encoder_input[i:i+512]) , np.array(label[i:i+512])

    def count_number_trainable_params(self, variable_scope):
        """
        Counts the number of trainable variables.
        """
        tot_nb_params = 0
        for trainable_variable in variable_scope:
            shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
            print(shape)
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

