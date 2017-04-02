from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy
from base import Model
import pdb

class tcNet(Model):
	def __init__(self, sess, num_classifier, num_class = 4716):
		self.sess = sess
		self.num_class = num_class
		self.num_classifier = num_classifier
		#[batch_size, max_frame_size, 1024]
		self.frame_features = tf.placeholder(tf.float32, shape = [None, None, 1024])
		#[batch_size, num_class]
		self.labels_rough = tf.placeholder(tf.float32, shape = [None, self.num_classifier])
		#[batch_size, num_class]
		self.labels_fine = tf.placeholder(tf.float32, shape = [None, self.num_class])
		#[batch_size]
		self.batch_lengths = tf.placeholder(tf.int32, shape = [None])
	
	#kernel_sizes: [[w, channel, pool:<strides/None>]]
	def build(self, rnn_hidden_size, cnn_kernel_sizes, cls_feature_dim, phase,
			  lr = 1e-4, weight_decay = 1e-4, dropout_ratio = 0.5, train = True):
		#define model hyperparameters
		self.rnn_hidden_size = rnn_hidden_size
		self.cnn_kernel_sizes = cnn_kernel_sizes
		self.cls_feature_dim = cls_feature_dim
		self.phase = phase
		self.lr = lr
		self.opt = tf.train.AdamOptimizer(self.lr)
		self.weight_decay = weight_decay
		self.dropout_ratio = dropout_ratio
		self.train = train
		if not self.train:
			self.dropout_ratio = 0

		#######################################################################
		#######################################################################

		#define rnn
		self.cell = tf.contrib.rnn.GRUCell(self.rnn_hidden_size)
		self.cell_dropout = tf.contrib.rnn.DropoutWrapper(self.cell, output_keep_prob = 1 - self.dropout_ratio)
		self.bi_features, _ = tf.nn.bidirectional_dynamic_rnn(
			cell_fw = self.cell_dropout,
			cell_bw = self.cell_dropout,
			dtype = tf.float32,
			sequence_length = self.batch_lengths,
			inputs = self.frame_features)
		#[batch_size, max_frame_size, rnn_hidden_size]
		self.features = self.bi_features[0] + self.bi_features[1]

		#[batch_size, 1, max_frame_size, rnn_hidden_size]
		self.cnn_inputs = tf.expand_dims(self.features, 1)
		self.cnn_layers = [self.cnn_inputs]
		
		#define cnn
		for idx, k_size in enumerate(cnn_kernel_sizes):
			if idx == 0:
				kernel_shape = [1, k_size[0], self.rnn_hidden_size, k_size[1]]
			else:
				kernel_shape = [1, k_size[0], cnn_kernel_sizes[idx - 1][1], k_size[1]]
			cnn = self.conv_layer(self.cnn_layers[-1], kernel_shape, 1e-3, 'conv%d' % idx)
			cnn_relu = tf.nn.relu(cnn)
			if k_size[2] is not None:
				pool_kernel_shape = [1, 1, k_size[2], 1]
				cnn_relu = tf.nn.max_pool(cnn_relu, pool_kernel_shape, pool_kernel_shape, 'VALID', name = 'conv%d_pool' % idx)
			self.cnn_layers.append(cnn_relu)
		
		#[batch_size, feature_dim]
		self.cnn_output = tf.reduce_max(self.cnn_layers[-1], axis = [1, 2])
		self.cls_features = self.fc_layer(self.cnn_output,
			[self.cnn_output.get_shape().as_list()[1], self.cls_feature_dim], 1e-4, 'cls_feature')
		self.cls_features_relu = tf.nn.relu(self.cls_features)
		self.cls_level1 = self.fc_layer(self.cls_features_relu, 
				[self.cls_feature_dim, self.num_classifier], 1e-4, 'cls_level1')
		if self.phase == 'phase1':
			self.cls_loss = tf.nn.softmax_cross_entropy_with_logits(labels = self.labels_rough, 
																	logits = self.cls_level1)
			self.wd = tf.add_n(tf.get_collection('all_weight_decay'), name = 'weight_decay_summation')
			self.loss = tf.reduce_mean(self.cls_loss) + self.wd
			self.minimize = self.opt.minimize(self.loss)
		elif self.phase == 'phase2' or self.phase == 'phase3':
			self.cls_level1_prob = tf.expand_dims(tf.transpose(tf.nn.softmax(self.cls_level1)), -1)
			self.classifiers = tf.Variable(tf.random_normal(
				[self.num_classifier, self.cls_feature_dim, self.num_class]), 
				stddev = 1e-3, name = 'fine_classifiers')
			#add weight decay for this classifier variable
			classifier_wd = tf.multiply(self.weight_decay,
				tf.nn.l2_loss(self.classifiers), name = 'classifier_weight_decay')
			tf.add_to_collection('all_weight_decay', classifier_wd)	

			self.pre_cls_level2 = tf.expand_dims(self.cls_features_relu, 0)
			self.pre_cls_level2 = tf.tile(self.pre_cls_level2, [self.num_classifier, 1, 1])
			self.cls_level2 = tf.matmul(self.pre_cls_level2, self.classifiers)
			self.cls_level2_prob = tf.nn.sigmoid(self.cls_level2)
			self.avg_cls_level2 = tf.multiply(self.cls_level1_prob, self.cls_level2_prob)
			self.cls = tf.reduce_sum(self.avg_cls_level2, 0)
			self.cls_recover = - tf.log(1 / self.cls - 1)
			self.cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.labels_fine,
																	logits = self.cls_recover)
			if self.phase == 'phase2':
				self.wd = classifier_wd
				self.loss = self.wd + self.cls_loss
				self.minimize = self.opt.minimize(self.loss, var_list = [self.classifiers])
			else:
				self.wd = tf.add_n(tf.get_collection('all_weight_decay'), name = 'weight_decay_summation')
				self.loss = self.wd + self.cls_loss
				self.minimize = self.opt.minimize(self.loss)
		else:
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
			return tf.nn.bias_add(tf.matmul(input, W), bias)
	
def main():
	sess = tf.Session()
	net = tcNet(sess, 34)
	#[stride, channel, pool_stride]
	cnn_kernels = [[3, 2048, None],
				   [3, 4096, 2],
				   [3, 4096, None],
				   [3, 4096, 2],
				   [3, 2048, 2]]
	with tf.device('/gpu: 0'):
		net.build(1024, cnn_kernels, 4096, 'phase1', 1e-4)


if __name__ == '__main__':
	main()
