from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy
from base import Model
import pdb

def lrelu(x, leak = 0.1, name = "lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * abs(x)

class tcNet(Model):
	def __init__(self, sess, num_classifier = 25, num_class = 4716):
		self.sess = sess
		self.num_class = num_class
		self.num_classifier = num_classifier
		#[batch_size, max_frame_size, 1024]
		self.frame_features = tf.placeholder(tf.float32, shape = [None, 300, 1024])
		#[batch_size, num_class]
		self.labels_rough = tf.placeholder(tf.float32, shape = [None, self.num_classifier])
		#[batch_size]
		self.labels_rough_factor = tf.placeholder(tf.float32, shape = [None])
		#[batch_size, num_class]
		self.labels_fine = tf.placeholder(tf.float32, shape = [None, self.num_class])
		#[batch_size, num_class]
		self.labels_fine_factor = tf.placeholder(tf.float32)
		#[batch_size]
		self.batch_lengths = tf.placeholder(tf.int32, shape = [None])
		self.new_varlist = []
		self.varlist = []
		self.optimize_varlist = []
	
	#kernel_sizes: [[w, channel, pool:<strides/None>, std]]
	def build(self, rnn_hidden_size, cnn_kernel_sizes, cls_feature_dim, phase,
			  lr, weight_decay, dropout_ratio = 0.4, train = True):
		#define model hyperparameters
		self.rnn_hidden_size = rnn_hidden_size
		self.cnn_kernel_sizes = cnn_kernel_sizes
		self.cls_feature_dim = cls_feature_dim
		self.phase = phase
		self.lr = lr
		self.opt_1 = tf.train.AdamOptimizer(self.lr[0])
		self.opt_2 = tf.train.AdamOptimizer(self.lr[1])
		self.opt_3 = tf.train.AdamOptimizer(self.lr[2])
		self.weight_decay = weight_decay
		self.dropout_ratio = dropout_ratio
		self.train = train
		if not self.train:
			self.dropout_ratio = 0
		#######################################################################
		#######################################################################
		#define rnn
		self.cell = tf.contrib.rnn.LSTMCell(self.rnn_hidden_size,
			initializer = tf.random_normal_initializer(stddev = 1e-1))
		self.cell_dropout = tf.contrib.rnn.DropoutWrapper(self.cell, output_keep_prob = 1 - self.dropout_ratio)
		self.bi_features, _ = tf.nn.bidirectional_dynamic_rnn(
			cell_fw = self.cell_dropout,
			cell_bw = self.cell_dropout,
			dtype = tf.float32,
			sequence_length = self.batch_lengths,
			inputs = self.frame_features)
		#[batch_size, max_frame_size, rnn_hidden_size]
		self.rnn_features = tf.add(self.bi_features[0], self.bi_features[1])

		#define cnn
		#[batch_size, 1, max_frame_size, rnn_hidden_size]
		self.cnn_inputs = tf.expand_dims(self.rnn_features, 1)
		self.cnn_layers = [self.cnn_inputs]
		for idx, k_size in enumerate(cnn_kernel_sizes):
			if idx == 0:
				kernel_shape = [1, k_size[0], self.rnn_hidden_size, k_size[1]]
			else:
				kernel_shape = [1, k_size[0], cnn_kernel_sizes[idx - 1][1], k_size[1]]
			cnn = self.conv_layer(self.cnn_layers[-1], kernel_shape, k_size[3], 'conv%d' % idx)
			cnn_relu = lrelu(cnn)
			if k_size[2] is not None:
				pool_kernel_shape = [1, 1, k_size[2], 1]
				cnn_relu = tf.nn.max_pool(cnn_relu, pool_kernel_shape, pool_kernel_shape, 'VALID', name = 'conv%d_pool' % idx)
			self.cnn_layers.append(cnn_relu)
		
		#[batch_size, feature_dim]
		cnn_shapes = self.cnn_layers[-1].get_shape().as_list()
		self.cnn_output = tf.reshape(self.cnn_layers[-1], [-1, cnn_shapes[1] * cnn_shapes[2] * cnn_shapes[3]])
		
		#############################
		#####  fine prediction  #####
		#############################
		self.cls_features_1, _ = self.fc_layer(self.cnn_output,
			[self.cnn_output.get_shape().as_list()[-1], self.cls_feature_dim[0]], 0.01, 'cls_feature_1')
		self.cls_features_1_relu = tf.nn.relu(tf.nn.dropout(self.cls_features_1, 1 - self.dropout_ratio))
		self.cls_features_2, _ = self.fc_layer(self.cls_features_1_relu,
			[self.cls_feature_dim[0], self.cls_feature_dim[1]], 0.01, 'cls_feature_2')
		self.cls_features_2_relu = tf.nn.relu(tf.nn.dropout(self.cls_features_2, 1 - self.dropout_ratio))
		self.cls_features_3, _ = self.fc_layer(self.cls_features_2_relu,
			[self.cls_feature_dim[1], self.cls_feature_dim[2]], 1e-1, 'cls_feature_3')
		self.cls_features_3_relu = tf.nn.relu(tf.nn.dropout(self.cls_features_3, 1 - self.dropout_ratio))
		self.cls_features_4, _ = self.fc_layer(self.cls_features_3_relu,
			[self.cls_feature_dim[2], self.cls_feature_dim[3]], 1e-2, 'cls_feature_4')
		self.cls_features_4_relu = tf.nn.relu(tf.nn.dropout(self.cls_features_4, 1 - self.dropout_ratio))
		
		self.classifiers = tf.Variable(tf.random_normal(
			[self.num_classifier, self.cls_feature_dim[3], self.num_class],
			stddev = 1e-2, name = 'fine_classifiers_2'))
		#add weight decay for this classifier variable
		classifier_wd = tf.multiply(self.weight_decay,
			tf.nn.l2_loss(self.classifiers), name = 'classifier_weight_decay')
		tf.add_to_collection('all_weight_decay', classifier_wd)
		self.cls_features_copy = tf.tile(self.cls_features_4_relu, [self.num_classifier, 1, 1])
		self.cls_level2 = tf.matmul(self.cls_features_copy, self.classifiers)
		self.cls_level2_prob = tf.nn.sigmoid(self.cls_level2)
		#merge two levels
		self.cls_level1_prob = tf.expand_dims(tf.transpose(self.cls_level1_prob), -1)
		self.avg_cls_level2 = tf.multiply(self.cls_level1_prob, self.cls_level2_prob)
		self.cls = tf.reduce_sum(self.avg_cls_level2, 0)
		self.cls_loss = self.xentropy_loss(self.cls, self.labels_fine, self.labels_fine_factor)

		self.wd = tf.add_n(tf.get_collection('all_weight_decay'), name = 'weight_decay_summation')
		self.loss = self.wd + self.cls_loss 
		self.minimize_fine = self.opt_2.minimize(self.loss, var_list = list(set(tf.global_variables()) - set(self.phase1_varlist)))
		self.minimize = tf.group(self.opt_1.minimize(self.loss, var_list = self.phase1_varlist), self.minimize_rough, self.minimize_fine)
		#self.phase2_varlist = list(set(tf.global_variables()) - set(self.phase1_varlist))
		elif self.phase != 'phase1':
			print('Wrong Phase number: <phase1|phase2|phase3>')
			assert(0)
		

	def conv_layer(self, input, kernel_size, std, name):
		with tf.variable_scope(name):
			init_filt = tf.random_normal_initializer(mean = 0.0, stddev = std)
			init_bias = tf.constant_initializer(value = 0.0, dtype = tf.float32)

			filt = tf.get_variable(name = 'weights', initializer = init_filt,
				shape = kernel_size, dtype = tf.float32)
			weight_decay = tf.multiply(self.weight_decay, tf.nn.l2_loss(filt), name = 'weight_decay')
			tf.add_to_collection('all_weight_decay', weight_decay)
			bias = tf.get_variable(name = 'bias', initializer = init_bias, shape = [kernel_size[-1]])
			return tf.nn.conv2d(input, filt, [1, 1, 1, 1], 'SAME')

	def fc_layer(self, input, shape, std, name):
		with tf.variable_scope(name):
			init_W = tf.random_normal_initializer(mean = 0.0, stddev = std)
			init_b = tf.constant_initializer(value = 0.0, dtype = tf.float32)

			W = tf.get_variable(name = 'weights', initializer = init_W,
				shape = shape, dtype = tf.float32)
			weight_decay = tf.multiply(self.weight_decay, tf.nn.l2_loss(W), name = 'weight_decay')
			tf.add_to_collection('all_weight_decay', weight_decay)
			bias = tf.get_variable(name = 'bias', initializer = init_b, shape = [shape[-1]])
			return tf.nn.bias_add(tf.matmul(input, W), bias), [W, bias]

	def xentropy_loss(self, predictions, labels, bias):
		with tf.name_scope("loss_xent"):
			epsilon = 10e-6
			float_labels = tf.cast(labels, tf.float32)
			cross_entropy_loss = float_labels * tf.log(predictions + epsilon) +\
				(1 - float_labels) * tf.log(1 - predictions + epsilon)
			cross_entropy_loss = tf.negative(cross_entropy_loss) * bias
			return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))

	def softmax_loss(self, predictions, labels, bias):
		with tf.name_scope("loss_softmax"):
			epsilon = 10e-8
			float_labels = tf.cast(labels, tf.float32)
			# l1 normalization (labels are no less than 0)
			label_rowsum = tf.maximum(
				tf.reduce_sum(float_labels, 1, keep_dims=True),
				epsilon)
			norm_float_labels = tf.div(float_labels, label_rowsum)
			softmax_outputs = tf.nn.softmax(predictions)
			softmax_loss = tf.negative(tf.reduce_sum(
				tf.multiply(norm_float_labels, tf.log(softmax_outputs)), 1))
			softmax_loss = softmax_loss * bias
			return tf.reduce_mean(softmax_loss)
