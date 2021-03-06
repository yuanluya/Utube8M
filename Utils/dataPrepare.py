import tensorflow as tf
import pdb
import numpy as np
import scipy.stats as scp
from random import shuffle
import os
import json
import sys
sys.path.append('../Evaluation')
from eval_util import EvaluationMetrics

class tfReader:
	def __init__(self, sess, record_dir, mode, rough_bias, max_video_len = 300, num_classifiers = 25,
				 num_features = 4716, feature_vec_len = 1024, pad_batch_max = False):

		self.small2big = json.load(open('../Utils/small2big.json'))
		self.evaluator = EvaluationMetrics(num_features, 1)
		self.evaluator_rough = EvaluationMetrics(num_classifiers, 1)
		self.sess = sess
		self.record_dir = os.path.join(record_dir, mode)
		self.mode = mode
		self.rough_bias = rough_bias
		self.max_video_len = max_video_len
		self.reader = tf.TFRecordReader()
		self.num_classifiers = num_classifiers
		self.num_features = num_features
		self.feature_vec_len = feature_vec_len
		self.pad_batch_max = pad_batch_max

		self.record_names_all = [os.path.join(self.record_dir, f) for f in os.listdir(self.record_dir) if f[-8: ] == 'tfrecord']
		old_names = open('../Data/old_name.txt', 'r')
		self.record_names_old = old_names.readlines()
		self.record_names_old = [os.path.join(self.record_dir, name[0: -1]) for name in self.record_names_old if len(name) > 0]
		self.record_names = list(set(self.record_names_all) - set(self.record_names_old))
		shuffle(self.record_names)
		print('[DATA]Found %d records.' % len(self.record_names))
		filename_queue = tf.train.string_input_producer(self.record_names)
		_, serialized_example = self.reader.read(filename_queue)

		if self.mode != 'test':
			self.contexts, self.features = tf.parse_single_sequence_example(
				serialized_example,
				context_features={
					'video_id': tf.FixedLenFeature([], tf.string),
					'labels': tf.VarLenFeature(tf.int64)
				},
				sequence_features = {
					'rgb': tf.FixedLenSequenceFeature([], dtype = tf.string)
				})
		elif self.mode == 'test':
			self.contexts, self.features = tf.parse_single_sequence_example(
				serialized_example,
				context_features={
					'video_id': tf.FixedLenFeature([], tf.string),
				},
				sequence_features = {
					'rgb': tf.FixedLenSequenceFeature([], dtype = tf.string)
				})
		else:
			print('Wrong Mode Return')
			assert(0)
		self.frame_features = tf.reshape(tf.cast(
			tf.decode_raw(self.features['rgb'], tf.uint8), tf.float32), [-1, self.feature_vec_len])

	def fetch(self, batch_size):
		batch_data = []
		for _ in range(batch_size):
			one_data = {}
			if self.mode != 'test':
				[frame_features, labels, v_ids] = self.sess.run([self.frame_features, 
																 self.contexts['labels'], 
																 self.contexts['video_id']])
				one_data['labels'] = labels[1]
			else: 
				[frame_features, v_ids] = self.sess.run([self.frame_features, self.contexts['video_id']])
			one_data['features'] = frame_features
			one_data['ids'] = v_ids
			batch_data.append(one_data)
		return self.preProcess(batch_data)

	def preProcess(self, batch_data):
		batch_size = len(batch_data)
		pad_feature = np.empty((batch_size, self.max_video_len, self.feature_vec_len))
		original_len = []
		for i, data in enumerate(batch_data):
			original_len.append(min(data['features'].shape[0], self.max_video_len))
			cut_length = self.max_video_len -  data['features'].shape[0]
			if cut_length >= 0:
				tmp = np.append(data['features'], np.zeros((cut_length, self.feature_vec_len)), 0)
				pad_feature[i] = tmp
			else:
				pad_feature[i] = data['features'][0: self.max_video_len, :]

		real_batch_data = {'pad_feature': pad_feature,
						   'original_len': np.array(original_len),
						   'video_ids': [b['ids'] for b in batch_data]}

		if self.mode != 'test':
			labels_fine = np.zeros((batch_size, self.num_features))
			labels_rough, labels_rough_factor = self.fine2rough(batch_data)
			for i, data in enumerate(batch_data):
				labels_fine[i][data['labels']] = 1

			labels_fine_factor = self.generate_factor(labels_fine)
			real_batch_data.update({'labels_rough': labels_rough,
							   		'labels_rough_factor': labels_rough_factor,
							   		'labels_fine': labels_fine, 
							   		'labels_fine_factor': 1.0#labels_fine_factor
							   		})
			
		return real_batch_data
	
	def fine2rough(self, batch_data):
		label_rough = np.zeros([len(batch_data), self.num_classifiers])
		for b, data in enumerate(batch_data):
			temp_rough = [self.small2big[str(s)] for s in data['labels']]
			label_rough[b, temp_rough] = 1
		label_rough_factor = self.generate_factor(label_rough, 1)
		return label_rough, 1.0#label_rough_factor

	#calculate label factor
	#normalize positive and negative within each class(column) 
	#pos_neg_ratio: how much MORE negative samples are weighted
	def generate_factor(self, labels_fine, pos_neg_ratio = 1):
		denominator_one = np.sum(labels_fine ,axis = 0)
		denominator_zero = np.ones(labels_fine.shape[1]) * labels_fine.shape[0] - denominator_one
		denominator_one = 1 / denominator_one
		denominator_zero = 1 / denominator_zero
		denominator_one[np.argwhere(np.isinf(denominator_one))] = 1
		denominator_zero[np.argwhere(np.isinf(-denominator_zero))] = 1
		fine_factor_one = labels_fine * denominator_one
		fine_factor_zero = labels_fine - 1
		fine_factor_zero = fine_factor_zero * denominator_zero
		labels_fine_factor = fine_factor_one - pos_neg_ratio * fine_factor_zero
		return labels_fine_factor


if __name__ == '__main__':
	main()
