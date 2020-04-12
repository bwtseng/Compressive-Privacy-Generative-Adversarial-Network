import numpy as np
import os 
import time
import random
import math
import data as dataset
import matplotlib.pyplot as plt 
import pandas as pd 
from keras.datasets import cifar10
import tensorflow as tf 
import tensorflow.contrib.layers as ly 
from scipy.misc import imrotate ,imread ,imsave, imresize
from keras.datasets import cifar10
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

tf.set_random_seed(9)
np.random.seed(9)
plt.switch_backend('agg')

class shakenet:

	def __init__(self,arg):

		tf.set_random_seed(9)
		np.random.seed(9)

		self.arg = arg
		self.batch_size = 128
		self.init_learning_rate = 0.2
		self.num_epochs = 1800

		self.output_g = []
		self.output_c = []
		self.train_set , self.val_set = self.load_data()
		self.build_model()

		print("The number of multiplication and addtion : {} and {}.".format(self.g_multiplication,self.g_addition))
		print("The number of multiplication and addtion : {} and {}.".format(self.c_multiplication,self.c_addition))

		self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob, 1),tf.argmax(self.label_p, 1)), tf.float32))
		utility_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label_p, logits=self.logit))

		self.loss_r = tf.losses.mean_squared_error(self.image_p, self.up) 
		self.loss_g = utility_loss - tf.losses.mean_squared_error(self.image_p,self.up) 
		self.theta_r = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='reconstructor')
		self.theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='compressor')
		self.theta_c_up = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='utility_classifier/decoder')
		self.theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='utility_classifier/classifier')
		l2_reg_loss = tf.add_n([tf.nn.l2_loss(var) for var in self.theta_c])
		self.loss_c = utility_loss + 0.0001 * l2_reg_loss
		print('The numbers of parameters in variable_scope G are : {}'.format(self.count_number_trainable_params(self.theta_g)))
		print('The numbers of parameters in variable_scope R are : {}'.format(self.count_number_trainable_params(self.theta_c)))

		uti_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(uti_update):
            ### Be careful to tune the learning of each three subnetworks.
			self.g_op = tf.train.AdamOptimizer(0.001)
			self.g_opt = self.g_op.minimize(self.loss_g, var_list=self.theta_g)

			self.r_op = tf.train.AdamOptimizer(0.001)
			self.r_opt = self.r_op.minimize(self.loss_r, var_list=self.theta_r)

			self.c_op = tf.train.MomentumOptimizer(self.learning_rate_p, 0.9, use_nesterov=True)
			self.c_opt = self.c_op.minimize(self.loss_c,var_list = self.theta_c + self.theta_c_up)

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(max_to_keep=5)
		self.assign_op = [] 

	def preprocess(self,data):
		a=[]
		for i in  data: 
			temp = self.plot(i)
			temp = (temp/127.5)-1
			a.append(temp)
		return a

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

	def build_model(self):

		self.image_p = tf.placeholder(tf.float32,shape=[None, 32, 32, 3])
		self.label_p = tf.placeholder(tf.int64,shape=[None, 10])
		self.is_train = tf.placeholder(tf.bool)
		self.learning_rate_p = tf.placeholder(tf.float32)
		self.noise_p = tf.placeholder(tf.float32,shape=[None, 32, 32, 3])

		self.latent = self.residual_g(self.image_p)
		self.up = self.decoder_conv(self.latent)
		self.logit = self.utility_classifier(self.latent, 10)
		self.prob = tf.nn.softmax(self.logit)

	def bo_batch_norm(self,x, is_training, momentum=0.9, epsilon=0.00001):
		"""
		Add a new batch-normalization layer.
		:param x: tf.Tensor, shape: (N, H, W, C).
		:param is_training: bool, train mode : True, test mode : False
		:return: tf.Tensor.
		"""
		x = tf.layers.batch_normalization(x, momentum=momentum, epsilon=epsilon, training=is_training)
		return x

	def residual_g(self,image,reuse=False):
		stride = [1,1,1]
		filter_size = [3,3,3]

		self.g_addition = 0 
		self.g_multiplication = 0

		with tf.variable_scope('compressor') as scope:
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
			return x 

	def decoder_conv(self,latent,reuse=False):
		with tf.variable_scope('reconstructor') as scope:
			if reuse:
				scope.reuse_variables()
			dim = 32
			latent = ly.flatten(latent)
			latent = ly.fully_connected(latent, 4*4*64, activation_fn=tf.nn.relu)
			latent = self.bo_batch_norm(latent, self.is_train)
			latent = tf.reshape(latent, shape=[-1,4,4,64])
			upsample1 = ly.conv2d_transpose(latent, dim*4, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
			upsample2 = ly.conv2d_transpose(upsample1, dim*2, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
			upsample3 = ly.conv2d_transpose(upsample2, 3, kernel_size=3, stride=2, padding='SAME', activation_fn=tf.nn.tanh, weights_initializer=tf.random_normal_initializer(0, 0.02))
		return upsample3
	
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
				temp = bnorm1.get_shape().as_list()
				self.c_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
				self.c_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel) 

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
			temp = x.get_shape().as_list()
			self.c_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
			self.c_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel) 

			#self.c_multiplication +=1 
			#self.c_addition += 1 

		# relu2 - conv2 - batch_norm2 with stride = 1
		with tf.variable_scope('branch_conv_bn2'):
			x = tf.nn.relu(x)

			channel = x.get_shape().as_list()[-1]

			x = self.conv_layer_no_bias(x, 3, 1, output_filters) # stirde = 1

			temp = x.get_shape().as_list()
			self.c_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel)
			self.c_addition += (temp[1]*temp[2]*temp[3]*((3*3*channel-1)))

			x = self.batch_norm(x, is_training=self.is_train)

			temp = x.get_shape().as_list()
			self.c_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
			self.c_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel) 

		#### condition on the forward and backward need different flow chart.
		#### refer to https://stackoverflow.com/questions/36456436/how-can-i-define-only-the-gradient-for-a-tensorflow-subgraph/36480182#36480182
		x = tf.cond(self.is_train, lambda: x * random_backward + tf.stop_gradient(x * random_forward - x * random_backward) , lambda: x / num_branches)

		self.c_multiplication += (temp[1]*temp[2]*temp[3]) 

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
	
				temp = bn_path.get_shape().as_list()
				self.c_addition += (temp[1]*temp[2]*temp[3]*3*3*channel)
				self.c_multiplication += (temp[1]*temp[2]*temp[3]*3*3*channel) 

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

	def bias_variable(self,shape, value=1.0):
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
		c = list(zip(t_data,t_label))
		random.shuffle(c)
		t_data , t_label = zip(*c)
		#self.curr_learning_rate = self.init_learning_rate
		for i in range(0,le,128):
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
					yield np.array(self.batch_crop_image(t_data[i:])+self.batch_crop_image(t_data[:sup])), np.array(t_label[i:i+128]), np.array(self.batch_random_rotate_image(t_data[i:i+128])) , np.array(t_label[i:i+128]) ,aug
				elif aug == 0:
					yield np.array(t_data[i:i+128]) , np.array(t_label[i:i+128]), np.array(self.batch_mirror_image(t_data[i:i+128])) , np.array(t_label[i:i+128]) , aug
				elif aug == 2 : 
					yield np.array(t_data[i:i+128]) , np.array(t_label[i:i+128]), np.array(self.batch_crop_image(t_data[i:i+128])) , np.array(t_label[i:i+128]) , aug
				else :
					yield np.array(t_data[i:i+128]) , np.array(t_label[i:i+128]), np.array(t_data[i:i+128]) , np.array(t_label[i:i+128]) ,aug


	def train_next_batch(self,t_data,t_label,batch_size=128):
		le = len(t_data)
		epo = le//batch_size
		leftover = le - epo * batch_size
		sup = batch_size - leftover
		c = list(zip(t_data,t_label))
		random.shuffle(c)
		t_data , t_label = zip(*c)
		#self.curr_learning_rate = self.init_learning_rate
		for i in range(0,le,128):
			if i ==  (epo *batch_size) : 
				yield np.array(self.batch_crop_image(t_data[i:])+self.batch_crop_image(t_data[:sup])), np.concatenate((t_label[i:],t_label[:sup]),axis=0), np.array(self.batch_random_rotate_image(t_data[i:i+128])) , np.array(t_label[i:i+128]) 
			else : 
				yield np.array(self.batch_crop_image(t_data[i:i+128])) , np.array(t_label[i:i+128]), np.array(t_data[i:i+128]) , np.array(t_label[i:i+128]) 


	def plot_10slot(self, name):
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

	def predict(self,data):
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

			#no = np.random.normal(size=(_batch_size,32,32,3))
			no = np.random.laplace(size=(_batch_size,32,32,3))
			X , _ = data.next_batch(_batch_size,shuffle=False,augment=False,is_train=False)
			y_pred = self.sess.run(self.prob,feed_dict={self.image_p:X, self.is_train:False,self.noise_p:no})
			_y_pred.append(y_pred)
		_y_pred = np.concatenate(_y_pred, axis=0)
		y_true = data.labels
		acc = accuracy_score(np.argmax(y_true,1),np.argmax(_y_pred,1))
		return acc

	def train(self):
		citers = 25
		### PYWU server is 5   
		gen = 1
		train_size = self.train_set.num_examples
		num_steps_per_epoch = train_size // self.batch_size
		num_steps = self.num_epochs * num_steps_per_epoch
		self.curr_learning_rate = self.init_learning_rate
		count = 1
		""" 
		***************************************
		Server xist the pretrained model (ckpt file model_1798)
		***************************************
		""" 
		for i in range(num_steps):
			no = np.random.laplace(size=(self.batch_size, 32, 32, 3))
			X , y_true = self.train_set.next_batch(self.batch_size, shuffle=True, augment=True, is_train=True)
			feed_dict = {self.image_p:X, self.label_p:y_true, self.is_train:True, self.learning_rate_p:self.curr_learning_rate, self.noise_p:no}
			for _ in range(citers):
				_ = self.sess.run([self.r_opt],feed_dict=feed_dict)
			_, loss, y_pred = self.sess.run([self.c_opt,self.loss_c,self.prob],feed_dict=feed_dict)
			for _ in range(gen):
				_ = self.sess.run([self.g_opt],feed_dict=feed_dict)
			if (i+1) % num_steps_per_epoch ==0:
				#at_acc = self.predict(self.train_set)
				at_acc = accuracy_score(np.argmax(y_pred,1),np.argmax(y_true,1))
				av_acc = self.predict(self.val_set)
				print('Epoch {}   Training accuracy is {}, testing accuracy is {}, cur learning rate is {:.6f}.'.format(count,at_acc,av_acc,self.curr_learning_rate))
				self._update_learning_rate_cosine(i,num_steps)
				count += 1
				self.saver.save(self.sess,'model_'+str(count))


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

	def test(self):
		#self.saver.restore(self.sess,'cpgan_log/model_137')
		self.saver.restore(self.sess,self.arg.model_path)
		print('successfully restore')
		#print(len(self.sess.run(self.theta_g)))
		av_acc = self.predict(self.val_set)
		print('Testing accuracy is {}.'.format(av_acc))


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

	#def count_operation(self, shape):
	#    k , l = shape

	def get_nb_params_shape(self,shape):
		'''
		Computes the total number of params for a given shap.
		Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
		'''
		nb_params = 1
		for dim in shape:
			nb_params = nb_params*int(dim)
		return nb_params 
