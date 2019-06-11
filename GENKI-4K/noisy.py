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
from scipy.misc import imread, imresize, imsave



class syn_noise: 
    def __init__(self, arg):
        random.seed(9)
        np.random.seed(9)
        #tf.set_random_seed(9)

        self.arg = arg 
        self.path = self.arg.path
        #self.ori_dim = self.arg.ori_dim
        self.com_dim = 256
        self.noise_factor = self.arg.noise_term
        self.noise_scale = self.arg.noise_scale
        self.batch_size = self.arg.batch_size
        self.epo = self.arg.epoch
        self.t_data, self.t_label, self.v_data, self.v_label, t_data_n, v_data_n= self.load_data()
        self.n_t_data, self.n_v_data = self.noise_data(t_data_n, v_data_n)

        self.t_data = self.normalize(self.t_data)
        self.t_data = np.concatenate(self.t_data, axis=0)

        self.v_data = self.normalize(self.v_data)
        self.v_data = np.concatenate(self.v_data, axis=0)
        self.DNN()

    def noise_data(self, t_data, v_data):
        # Add noise to 0-255 value.

        t_data = (t_data + 1 )* 127.5 
        v_data = (v_data + 1 )* 127.5 

        t_data = t_data + self.noise_factor * np.random.laplace(0, self.noise_scale, size=(t_data.shape[0], 64, 64, 3))
        t_data = np.clip(t_data, 0, 255)
        v_data = v_data + self.noise_factor * np.random.laplace(0, self.noise_scale, size=(v_data.shape[0], 64, 64, 3))
        v_data = np.clip(v_data, 0, 255)
        t_data = (t_data/127.5) - 1
        v_data = (v_data/127.5) - 1
        return t_data, v_data

    def normalize(self, img_list): 

        temp = []
        for img in img_list:
            for i in range(3):
                img[:, :, i] = (img[:, :, i] - np.min(img[:,:, i]))/ (np.max(img[:, :, i]) - np.min(img[:, :, i]) + 0.00001) 
            temp.append(img)
        return temp 

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

        t_data_n = (np.concatenate(t_data, axis=0)/127.5) - 1
        #print(t_data.shape)
        t_label = np.array(t_label)

        v_data_n = (np.concatenate(v_data, axis=0)/127.5) - 1
        #print(v_data.shape)
        v_label = np.array(v_label)

        return t_data, t_label, v_data, v_label, t_data_n, v_data_n

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

            conv2 = ly.conv2d(conv1, 256, kernel_size=5, stride=1, padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            #conv2 = self.bo_batch_norm(conv2, self.is_train)
            conv2 = ly.max_pool2d(conv2, kernel_size=3, stride=2, padding='SAME')

            conv3 = ly.conv2d(conv2, 384, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            #conv3 = self.bo_batch_norm(conv3, self.is_train)
            #conv3 = ly.max_pool2d(conv3,kernel_size=3,stride=2,padding='SAME')

            conv4 = ly.conv2d(conv3, 384, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            #conv4 = self.bo_batch_norm(conv4, self.is_train)
            #conv4 = ly.max_pool2d(conv4,kernel_size=3,stride=2,padding='SAME')

            conv5 = ly.conv2d(conv4, 256, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            #conv5 = self.bo_batch_norm(conv5, self.is_train)
            conv5 = ly.max_pool2d(conv5, kernel_size=3, stride=2, padding='SAME')
            flat = ly.flatten(conv5)
            print(flat)

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


    def DNN(self):

        self.data_p = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
        self.label_p = tf.placeholder(tf.int64, shape=[None])
        self.one_hot = tf.one_hot(self.label_p, 2)
        self.noise_p = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
        self.dropout_rate = tf.placeholder(tf.float32)
        ### Tune different factor. 
        self.data_perturbed = tf.add(self.data_p, self.noise_factor * self.noise_p)

        #self.flat, self.logit = self.Alex_net(self.data_p, "Deep_based", "Classifier")
        self.flat, self.logit = self.LeNet(self.data_p, "Deep_based", "Classifier")

        self.prob = tf.nn.softmax(self.logit)

        scope = ["Deep_based", "Classifier"]
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot, logits=self.logit))

        theta = [] 
        for i in scope :
            theta += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = i)
        #theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = scope)
        op = tf.train.AdamOptimizer()
        self.opt = op.minimize(self.loss, var_list = theta)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()


    def train(self):
        epo = 1
        for _ in range(self.epo):
            print("Epoch {} starts.".format(epo))
            #for i, j in self.next_batch(self.n_t_data, self.t_label, self.batch_size, shuffle=True):
            for i, j in self.next_batch(self.t_data, self.t_label, self.batch_size, shuffle=True):

                sample = i.shape[0]
                no  = np.random.laplace(size=(sample, 64, 64, 3))
                feed_dict = {} 
                feed_dict[self.data_p] = i
                feed_dict[self.label_p] = j
                feed_dict[self.noise_p] = no 
                loss, _ = self.sess.run([self.loss, self.opt], feed_dict=feed_dict)

            epo+=1
        acc = self.compute_acc()
        mse = self.compute_mse()
        return acc, mse 

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
                yield np.array(t_data[i:]).reshape(-1, 64, 64, 3) , np.array(t_label[i:])
            else : 
                yield np.array(t_data[i: i+self.batch_size]).reshape(-1, 64, 64, 3), np.array(t_label[i: i+self.batch_size])


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
        x = x.reshape(64, 64, 3)
        '''
        x += 1 
        x *= 127.5
        '''
        return x 

    def sample_noise(self, sample_size):
        temp = [] 
        for i in range(sample_size):
            temp.append(np.random.laplace(size=(1, 64, 64, 3)))
        return np.concatenate(temp, axis=0)

    def compute_mse(self):

        ### modify to self.perterb data

        temp = []
        temp_1 = []
        '''
        for i, j in self.next_batch(self.n_v_data, self.v_label, self.batch_size):
            sample = i.shape[0]
            no  = np.random.laplace(size=(sample, 64, 64, 3))
            feed_dict = {} 
            feed_dict[self.data_p] = i
            feed_dict[self.label_p] = j
            feed_dict[self.noise_p] = self.sample_noise(sample)
            perturb = self.sess.run(self.data_perturbed, feed_dict = feed_dict)
            for k in range(len(perturb)):
                #temp.append(mean_squared_error(self.plot(i[k]).flatten(), self.plot(perturb[k]).flatten()))
                temp.append(mean_squared_error((i[k]).flatten(), (perturb[k]).flatten()))
                #temp_1.append(mean_squared_error((i[k]*255).flatten(), (i[k]*255+ self.noise_factor * np.random.laplace(size=(1, 28,28,1))).flatten()))
        '''

        for i in range(len(self.v_data)): 
            temp.append(mean_squared_error(self.plot(self.v_data[i]).flatten(), self.plot(self.n_v_data[i]).flatten()))
        mse = np.mean(temp)
        mse = np.mean(temp)

        #mse_1 = np.mean(temp_1)
        print("Perturbated data MSE: {}".format(mse))
        #print("Other MSE: {}".format(mse_1))

        imsave("/home/pywu/bowei/Genki/reconstruction/ori_1.png", self.plot(self.v_data[0]))
        imsave("/home/pywu/bowei/Genki/reconstruction/ori_2.png", self.plot(self.v_data[1]))

        imsave("/home/pywu/bowei/Genki/reconstruction/noise_1.png", self.plot(self.n_v_data[0]))
        imsave("/home/pywu/bowei/Genki/reconstruction/noise_2.png", self.plot(self.n_v_data[1]))
        return mse

    def compute_acc(self):

        pred = [] 

        #for i, j in self.next_batch(self.n_v_data, self.v_label, self.batch_size):
        for i, j in self.next_batch(self.v_data, self.v_label, self.batch_size):

            sample = i.shape[0]
            no  = np.random.laplace(size=(sample, 64, 64, 3))
            feed_dict = {} 
            feed_dict[self.data_p] = i
            feed_dict[self.label_p] = j
            feed_dict[self.noise_p] = no
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
        print("Perturbated data accuracy: {}".format(accuracy))
        print("Perturbated data accuracy (TPN): {}".format(acc))

        return accuracy






