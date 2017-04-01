import tensorflow as tf
import numpy
from base import Model
import pdb

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

class tcNet(Model):
	def __init__(self, sess):
		self.sess = tf.Session()
		#[batch_size, max_frame_size, 1024]
		self.frame_features = tf.placeholder(tf.float32, shape = [None, None, 1024])
		#[batch_size, num_class]
		self.labels = tf.placeholder(tf.float32, shape = [None, 4716])
		#[batch_size]
		self.batch_lengths = tf.placeholder(tf.float32, shape = [None])
	
	#kernel_sizes: [[w, channel, pool:<strides/None>]]
	def build(self, rnn_hidden_size, cnn_kernel_sizes, cls_feature_dim, 
			  weight_decay = 1e-4, dropout_ratio = 0.5, train = True):
		#define model hyperparameters
		self.rnn_hidden_size = rnn_hidden_size
		self.cnn_kernel_sizes = cnn_kernel_sizes
		self.cls_feature_dim = cls_feature_dim
		self.weight_decay = weight_decay
		self.dropout_ratio = dropout_ratio
		self.train = train
		if not self.train:
			self.dropout_ratio = 0

		#######################################################################
		#######################################################################

		#define rnn
		self.cell = tf.nn.rnn_cell.GRUCell(self.rnn_hidden_size)
		self.cell_dropout = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob = 1 - self.dropout_ratio)
		sefl.bi_features, _ = tf.nn.bidirectional_dynamic_rnn(
			cell_fw = self.cell_dropout,
			cell_bw = self.cell_dropout,
			dtype = tf.float64,
			sequence_length = self.batch_lengths,
			inputs = self.frame_features)
		#[batch_size, max_frame_size, rnn_hidden_size]
		self.features = self.bi_features[0] + self.bi_features[1]

		#[batch_size, 1, max_frame_size, rnn_hidden_size]
		self.cnn_inputs = tf.expand_dim(self.features, 1)
		self.cnn_layers = [self.cnn_inputs]
		
		#define cnn
		for idx, k_size in enumerate(cnn_kernel_sizes):
			if idx == 0:
				kernel_shape = [1, k_size[0], self.rnn_hidden_size, k_size[1]]
			else:
				kernel_shape = [1, k_size[0], cnn_kernel_sizes[idx - 1][1], k_size[1]]
			cnn = self.conv_layer(self.cnn_layers[-1], kernel_shape, 1e-3, 'conv%d' % idx)
			cnn_relu = tf.nn.relu(cnn)
			if k_size[2] is None:
				pool_kernel_shape = [1, 1, k_size[2], 1]
				cnn_relu = tf.nn.max_pool(cnn_relu, pool_kernel_shape, pool_kernel_shape, 'VALID', name = 'conv%d_pool' % idx)
			self.cnn_layers.append(cnn_relu)
		
		#[batch_size, feature_dim]
		self.cnn_output = tf.reduce_max(self.cnn_layers[-1], axis = [1, 2])
		self.cls_features = self.fc_layer(self.cnn_output,
			[self.cnn_output.get_shape().as_list()[1], self.cls_feature_dim], 1e-4, 'cls_feature')
		self.cls = self.fc_layer(self.cls_features, [self.cls_feature_dim, 4716], 1e-4, 'cls_feature')
		
		self.wd = tf.add_n(tf.get_collection('all_weight_decay'), name = 'weight_decay_summation')
		self.cls_loss = tf.div(losses.hinge_loss(self.labels, self.cls), self.labels.get_shape().as_list()[0])
		self.loss = self.cls_loss + self.wd 

	def conv_layer(self, input, kernel_size, std, name):
		with tf.variable_scope(name):
			init_filt = tf.random_normal_initializer(mean = 0.0, stddev = std)
			init_bias = tf.constant_initializer(value = 0.0, dtype = tf.float32)

			filt = tf.get_variable(name = 'weights', initializer = init_filt,
				shape = kernel_size, dtype = tf.float32)
			weight_decay = tf.multiply(self.weight_decay, tf.nn.l2_loss(filt), name = 'weight_decay')
			tf.add_to_collection('all_weight_decay', weight_decay)
			bias = tf.get_variable(name = 'bias', initializer = init_bias)
			return tf.nn.conv2d(input, filt, [1, 1, 1, 1], 'SAME')

	def fc_layer(self, input, shape, std, name):
		with tf.variable_scope(name):
			init_W = tf.random_normal_initializer(mean = 0.0, stddev = std)
			init_b = tf.constant_initializer(value = 0.0, dtype = tf.float32)

			W = tf.get_variable(name = 'weights', initializer = init_W,
				shape = shape, dtype = tf.float32)
			weight_decay = tf.multiply(self.weight_decay, tf.nn.l2_loss(W), name = 'weight_decay')
			tf.add_to_collection('all_weight_decay', weight_decay)
			bias = tf.get_variable(name = 'bias', initializer = init_b)
			return tf.nn.bias_add(tf.matmul(input, W), bias)
