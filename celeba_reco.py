import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import h5py 
from functools import reduce
import tensorflow as tf
import tensorflow.contrib.layers as ly 
import time
from scipy.misc import imread, imresize
from sklearn.model_selection import train_test_split
import random
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd 
from sklearn.metrics import mean_squared_error

plt.switch_backend('agg')
tf.set_random_seed(9)
class cpgan: 
    def __init__(self,args):
        self.arg = args
        #self.emb = np.load('embedding.npy')
        all_data , all_label = self.load_data()
        self.t_data ,self.te_data ,self.t_label ,self.te_label = train_test_split(all_data,all_label,test_size=0.2,random_state=9)
        #self.t_data ,self.v_data ,self.t_label ,self.v_label = train_test_split(t_data,t_label,test_size=0.2,random_state=9)

        self.build_model()

        #self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob,1),self.label_p),tf.float32))

        #utility_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot,logits=self.classifier))

        self.loss_r = tf.losses.mean_squared_error(self.image_p,self.up) 
        #self.loss_c = utility_loss
        #self.loss_g = utility_loss - tf.losses.mean_squared_error(self.image_p,self.up) 

        self.theta_r = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='reconstructor')
        self.theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='compressor')
        #self.theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='utility_classifier')

        #self.g_opt = tf.train.AdamOptimizer(0.001).minimize(self.loss_g,var_list=self.theta_g)
        #self.c_opt = tf.train.AdamOptimizer(0.001).minimize(self.loss_c,var_list=self.theta_c)
        self.r_opt = tf.train.AdamOptimizer(0.001).minimize(self.loss_r,var_list=self.theta_r)#+self.theta_g)


        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
        self.assign_op = []
        weight_g = np.load('single_celebA/g.npy')
        for i in range(len(self.theta_g)):
            self.assign_op.append(tf.assign(self.theta_g[i],weight_g[i]))
        #self.t_emb , self.te_emb = self.get_emb()
        
    def load_data(self):
        temp =list(np.sort(os.listdir(self.arg.path)))
        img = []
        count = 0
        for i in temp : 
            img.append(os.path.join(self.arg.path,i))
        lab = []
        temp = pd.read_csv('All_label.csv')[[self.arg.attribute]].values.reshape(-1)
        for i in temp : 
            if i == -1:
                lab.append(0)
            else : 
                lab.append(1)
        print(len(lab))
        return img , lab 

    def preprocess(self, path_list):
        img = []
        #print(path_list[0])
        for path in path_list:
            #print(path)
            im = imread(path)
            im = im/127.5 
            im = im -1 
            #im  = im /255 
            img.append(im)
        return np.array(img)


    def get_emb(self):
        temp = []
        for j,k in self.next_batch(self.t_data,self.t_label):
            b = k.shape[0]
            no = np.random.normal(size=(b,64,64,3))
            temp.append(self.sess.run(self.latent_no,feed_dict={self.image_p:j.reshape(b,64,64,3),self.noise_p:no ,self.keep_prob:1}))

        temp_1 = []

        for j,k in self.next_batch(self.te_data,self.te_label):
            b = k.shape[0]
            no = np.random.normal(size=(b,64,64,3))
            temp_1.append(self.sess.run(self.latent_no,feed_dict={self.image_p:j.reshape(b,64,64,3),self.noise_p:no ,self.keep_prob:1}))
        return temp , temp_1

    def deep_g(self,image,reuse=False):

        dim = 32
        with tf.variable_scope('compressor') as scope:
            if reuse : 
                scope.reuse_variables()
            conv1 = ly.conv2d(image,64,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer(),normalizer_fn=ly.batch_norm)
            conv2 = ly.conv2d(conv1,64,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer(),normalizer_fn=ly.batch_norm)
            pool2 = ly.max_pool2d(conv2,2,padding='SAME')
            dropout2 = tf.nn.dropout(pool2,keep_prob=self.keep_prob)

            conv3 = ly.conv2d(dropout2,128,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer(),normalizer_fn=ly.batch_norm)
            conv4 = ly.conv2d(conv3,128,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer(),normalizer_fn=ly.batch_norm)
            pool4 = ly.max_pool2d(conv4,2,padding='SAME')
            dropout4 = tf.nn.dropout(pool4,keep_prob=self.keep_prob)

            conv5 = ly.conv2d(dropout4,256,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer(),normalizer_fn=ly.batch_norm)
            conv6 = ly.conv2d(conv5,256,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer(),normalizer_fn=ly.batch_norm)
            conv7 = ly.conv2d(conv6,256,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer(),normalizer_fn=ly.batch_norm)
            pool7 = ly.max_pool2d(conv7,2,padding='SAME')
            dropout7 = tf.nn.dropout(pool7,keep_prob=self.keep_prob)

            conv8 = ly.conv2d(dropout7,512,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer(),normalizer_fn=ly.batch_norm)
            conv9 = ly.conv2d(conv8,512,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer(),normalizer_fn=ly.batch_norm)
            conv10 = ly.conv2d(conv9,512,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer(),normalizer_fn=ly.batch_norm)
            pool10 = ly.max_pool2d(conv10,2,padding='SAME')
            dropout10 = tf.nn.dropout(pool10,keep_prob=self.keep_prob)

            conv11 = ly.conv2d(dropout10,512,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer(),normalizer_fn=ly.batch_norm)
            conv12 = ly.conv2d(conv11,512,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer(),normalizer_fn=ly.batch_norm)
            conv13 = ly.conv2d(conv12,512,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer(),normalizer_fn=ly.batch_norm)
            pool13 = ly.max_pool2d(conv13,2,padding='SAME')
            dropout13 = tf.nn.dropout(pool13,keep_prob=self.keep_prob)
            flat = ly.flatten(dropout13)
            flat = ly.fully_connected(flat,4096,activation_fn =tf.nn.relu,weights_initializer=tf.random_normal_initializer(0, 0.02))
            return flat


    def generator_conv(self,image,reuse=False):
        dim = 32
        with tf.variable_scope('compressor') as scope:
            if reuse : 
                scope.reuse_variables()
            conv1 = ly.conv2d(image,dim*1,kernel_size=3,stride=2,padding='SAME',activation_fn=tf.nn.leaky_relu,weights_initializer=tf.random_normal_initializer(0, 0.02))
            conv1 = self.bo_batch_norm(conv1, self.is_train)
            conv2 = ly.conv2d(conv1,dim*2,kernel_size=3,stride=2,padding='SAME',activation_fn=tf.nn.leaky_relu,weights_initializer=tf.random_normal_initializer(0, 0.02))
            conv2 = self.bo_batch_norm(conv2, self.is_train)
            conv3 = ly.conv2d(conv2,dim*4,kernel_size=3,stride=2,padding='SAME',activation_fn=tf.nn.leaky_relu,weights_initializer=tf.random_normal_initializer(0, 0.02))
            conv3 = self.bo_batch_norm(conv3, self.is_train)
            conv4 = ly.conv2d(conv3,dim*8,kernel_size=3,stride=2,padding='SAME',activation_fn=tf.nn.leaky_relu,weights_initializer=tf.random_normal_initializer(0, 0.02))
            conv4 = self.bo_batch_norm(conv4, self.is_train)
            #latent = ly.fully_connected(tf.reshape(conv4,shape=[-1,4*4*dim*8]),4096,activation_fn=tf.nn.leaky_relu)
            latent = ly.fully_connected(tf.reshape(conv4,shape=[-1,7*7*dim*8]),4096,activation_fn=tf.nn.leaky_relu)
            #print(conv4)
        return latent 

    def decoder_conv(self,latent,reuse=False):
        with tf.variable_scope('reconstructor') as scope:
            if reuse:
                scope.reuse_variables()
            latent = ly.fully_connected(latent,4*4*256,activation_fn =tf.nn.leaky_relu,normalizer_fn=ly.batch_norm,weights_initializer=tf.random_normal_initializer(0, 0.02))
            latent = tf.reshape(latent,shape=[-1,4,4,256])
            dim = 32
            unsample1 = ly.conv2d_transpose(latent,dim*4,kernel_size=3,stride=2,padding='SAME',activation_fn=tf.nn.leaky_relu,normalizer_fn=ly.batch_norm,weights_initializer=tf.random_normal_initializer(0, 0.02))
            upsample2 = ly.conv2d_transpose(unsample1,dim*2,kernel_size=3,stride=2,padding='SAME',activation_fn=tf.nn.leaky_relu,normalizer_fn=ly.batch_norm,weights_initializer=tf.random_normal_initializer(0, 0.02))
            upsample5 = ly.conv2d_transpose(upsample2 ,dim*1,kernel_size=3,stride=2,padding='SAME',activation_fn=tf.nn.leaky_relu,normalizer_fn=ly.batch_norm,weights_initializer=tf.random_normal_initializer(0, 0.02))
            upsample6 = ly.conv2d_transpose(upsample5 ,3,kernel_size=3,stride=2,padding='SAME',activation_fn=tf.nn.tanh,weights_initializer=tf.random_normal_initializer(0, 0.02))
        return upsample6  

    def utility_classifier(self,latent,classes,reuse=False):
        with tf.variable_scope('utility_classifier') as scope:
            if reuse:
                scope.reuse_variables()
            #classifier = tf.layers.dense(latent,256,activation=tf.nn.relu)
            classifier= ly.fully_connected(latent,512,activation_fn=tf.nn.leaky_relu,weights_initializer=tf.random_normal_initializer(0, 0.02),normalizer_fn=ly.batch_norm)
            classifier = tf.nn.dropout(classifier,keep_prob = self.keep_prob)
            classifier= ly.fully_connected(classifier,512,activation_fn=tf.nn.leaky_relu,weights_initializer=tf.random_normal_initializer(0, 0.02),normalizer_fn=ly.batch_norm)
            #classifier = tf.nn.dropout(classifier,keep_prob = self.keep_prob)
            classifier= ly.fully_connected(classifier, classes, activation_fn=None)
        return classifier

    def build_model(self):
        tf.reset_default_graph()
        self.image_p = tf.placeholder(tf.float32,shape=(None,64,64,3))
        #self.label_p = tf.placeholder(tf.int64,shape=(None)) 
        #self.emb_p = tf.placeholder(tf.float32,shape=(None,512))
        self.noise_p = tf.placeholder(tf.float32,shape=(None,64,64,3))
        self.keep_prob = tf.placeholder(tf.float32)
        #self.latent_no = self.deep_g(self.image_p)
        self.noisy_img = tf.add(self.image_p,self.noise_p)
        #self.latent_no = self.deep_g(self.noisy_img)
        self.latent_no = self.generator_conv(self.noise_img,reuse=True)
        #self.latent_no = tf.add(latent,latent_n)
        #self.up = self.decoder_conv(self.emb_p,reuse=False)
        self.up = self.decoder_conv(self.latent_no)
        #self.classifier = self.utility_classifier(self.latent_no,2,reuse=False)
        #self.prob = tf.nn.softmax(self.classifier)    
        #self.one_hot = tf.one_hot(self.label_p,2)

    def compute_acc(self):
        acc_list = []
        for j , k in self.next_batch(self.te_data,self.te_label):
            b = k.shape[0]
            no = np.random.normal(size=(b,64,64,3))
            pred = self.sess.run(self.prob,feed_dict={self.image_p:j.reshape(b,64,64,3),self.label_p:k,self.noise_p:no,self.keep_prob:1})
            acc_list += list(np.argmax(pred,1))

        correct = []

        for i in range(len(acc_list)):
            if acc_list[i] == self.te_label[i]:
                correct.append(1)

        print(len(correct)/len(self.te_label))

    def compute_acc_test(self):
        acc_list = []
        for j , k in self.next_batch(self.te_data,self.te_label):
            b = k.shape[0]
            no = np.random.normal(size=(b,64,64,3))
            pred = self.sess.run(self.prob,feed_dict={self.image_p:j.reshape(b,64,64,3),self.label_p:k,self.noise_p:no,self.keep_prob:1})
            acc_list += list(np.argmax(pred,1))

        correct = []
        for i in range(len(acc_list)):
            if acc_list[i] == self.te_label[i]:
                correct.append(1)

        print(len(correct)/len(self.te_label))

    def plot(self,x):
        x = x - np.min(x)
        x /= np.max(x)
        x *= 255  
        x= x.astype(np.uint8)
        x = x.reshape(64,64,3)
        return x 

    def plot_10slot(self,name):
        #'0'*(4-len(name)) + str(name)
        j = np.array(self.te_data[:128])
        k = np.array([i for i in range(128)])
        no = np.random.normal(size=(128,64,64,3))
        uu = self.sess.run(self.latent_no,feed_dict={self.image_p:j.reshape(128,64,64,3),self.noise_p:no,self.keep_prob:1})
        #uu = np.array(self.emb[0:128]).reshape(-1,512)
        yy = self.sess.run(self.up,feed_dict={self.latent_no:uu,self.keep_prob:1})

        #plt.imshow(plot(j[2]))
        #plt.imshow(plot(yy[2]))

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
        plt.savefig('single_celebA/'+'0'*(4-len(str(name))) + str(name)+'.png')
        #plt.show()

    #def mse_computing(self,batch_img,batch_reco_img):


    def compute_reco_mse(self):
        error = []
        for i , j in self.next_batch(self.te_data,self.te_label):
            b = j.shape[0]
            no = np.random.normal(size=(b,64,64,3))
            up = self.sess.run(self.up,feed_dict={self.image_p:i.reshape(b,64,64,3),self.noise_p:no,self.keep_prob:1})
            for k in range(len(up)):
                error.append(mean_squared_error(self.plot(i[k]).flatten(),self.plot(up[k]).flatten()))
        print('Average MSE among all testing images is {}'.format(np.mean(error)))
        return np.mean(error)


    def train(self):
        acc_trace = [] 
        mse_trace = [] 
        epochs = 180
        #os.mkdir('Attacker_Male_4096_vgg')
        for i in range(epochs):
            citers = 25
            start = time.time()
            count = 0
            for j , k in self.next_batch(self.t_data,self.t_label):
                print(count)
                self.sess.run(self.assign_op)
                b = k.shape[0]
                no = np.random.normal(size=(b,64,64,3))
                d_loss , _= self.sess.run([self.loss_r,self.r_opt],feed_dict={self.image_p:j.reshape(b,64,64,3),self.noise_p:no,self.keep_prob:1})
            end = time.time()
            #merge = sess.run(merged,feed_dict={image_p:j.reshape(b,64,64,3),label_p:k,noise_p:no})
            print('{}/{} epochs , cost {} sec , the reconstruction_loss = {}.'.format(i+1,epochs,end-start,d_loss))
            a = self.compute_reco_mse()
            mse_trace.append(a)
            if i %30 == 0:
                #self.compute_acc()
                self.plot_10slot(i)
                #self.saver.save(self.sess,'new_ae/reco'+str(i))
                #np.save('Attacker_Male_4096_vgg/'+'r_'+str(i)+'.npy',self.sess.run(self.theta_r))
                np.save('single_celebA/mse_trace.npy',mse_trace)
    def test(self):
        import matplotlib.pyplot as plt 
        import numpy as np 
        temp = np.load('single_celebAmse_trace.npy')
        #print(temp)
        le = len(list(temp))
        le = [i for i in range(le)]
        plt.plot(le,temp)
        plt.title('Training curve of the attcker network(single_celebA)')
        #plt.yticks([0.5,0.6,0.7,0.8,0.9,1,1.1,1.2])
        plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2])
        #plt.yticks([0.6,0.7,0.8,0.9,1])
        #plt.yticks([0.2,0.22,0.24,0.26,0.28])
        plt.xlabel('Epochs')
        plt.ylabel('Mean square error')
        plt.savefig('single_celebA_mse_trace.png')

    def see(self):
        self.sess.run(self.assign_op)
        self.saver.restore(self.sess,'attacker/reco700')
        print(self.sess.run(self.theta_r))

    def train_next_batch(self,encoder_input,label,batch_size=512):
        le = len(encoder_input)
        epo = le//batch_size
        leftover = le - (epo*batch_size)
        c = list(zip(encoder_input,label))
        random.shuffle(c)
        temp , temp_1 = zip(*c)
        for i in range(0,le,512):
            if i ==  (epo *batch_size) : 
                #yield np.array(encoder_input[i:]+encoder_input[0:(batch_size-leftover)]) , np.concatenate((label[i:],label[0:(batch_size-leftover)]),axis=0)
                #yield np.array(temp[i:]) , np.array(temp_1[i:])
                yield self.preprocess(temp[i:]) , np.array(temp_1[i:])
            else : 
                #yield np.array(temp[i:i+512]) , np.array(temp_1[i:i+512])
                yield self.preprocess(temp[i:i+512]) , np.array(temp_1[i:i+512])

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


