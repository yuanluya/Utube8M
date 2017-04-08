import tensorflow as tf
import pdb
import numpy as np
import scipy.stats as scp
from random import shuffle
import os
import json

class tfReader:
	def __init__(self, sess, record_dir, mode, max_video_len = 300, num_classifiers = 25,
				 num_features = 4716, feature_vec_len = 1024, pad_batch_max = False):

		self.small2big = json.load(open('../Utils/small2big.json'))
		self.sess = sess
		self.record_dir = os.path.join(record_dir, mode)
		self.mode = mode
		self.max_video_len = max_video_len
		self.reader = tf.TFRecordReader()
		self.num_classifiers = num_classifiers
		self.num_features = num_features
		self.feature_vec_len = feature_vec_len
		self.pad_batch_max = pad_batch_max

		self.record_names = [os.path.join(self.record_dir, f) for f in os.listdir(self.record_dir) if f[-8: ] == 'tfrecord']
		shuffle(self.record_names)
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

			#calculate label_fine_factor
			denominator_one = np.sum(labels_fine ,axis = 0)
			denominator_zero = np.ones(self.num_features) * batch_size - denominator_one
			denominator_one = 1 / denominator_one
			denominator_zero = 1 / denominator_zero
			denominator_one[np.argwhere(np.isinf(denominator_one))] = 1
			denominator_zero[np.argwhere(np.isinf(-denominator_zero))] = 1
			fine_factor_one = labels_fine * denominator_one
			fine_factor_zero = labels_fine - 1
			fine_factor_zero = fine_factor_zero * denominator_zero
			labels_fine_factor = fine_factor_one - fine_factor_zero
			real_batch_data.update({'labels_rough': labels_rough,
							   		'labels_rough_factor': labels_rough_factor,
							   		'labels_fine': labels_fine, 
							   		'labels_fine_factor': labels_fine_factor
							   		})
			
		return real_batch_data
	
	def fine2rough(self, batch_data):
		label_rough = np.zeros([len(batch_data), self.num_classifiers])
		all_rough_labels = np.zeros(len(batch_data))
		for b, data in enumerate(batch_data):
			temp_rough = [self.small2big[str(s)] for s in data['labels']]
			rough_label = scp.mode(temp_rough)[0][0]
			all_rough_labels[b] = rough_label
			label_rough[b, rough_label] = 1
		
		dominate_class = int(scp.mode(rough_label)[0][0])
		label_rough_factor = np.ones(len(batch_data))
		label_rough_factor = label_rough_factor - 1.5 * (all_rough_labels == dominate_class)

		return label_rough, label_rough_factor 

if __name__ == '__main__':
	main()
