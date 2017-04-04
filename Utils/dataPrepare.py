import tensorflow as tf
import pdb
import numpy as np
import scipy.stats as scp
import json

class tfReader:
	def __init__(self, sess, record_names, num_classifiers = 25,
				 num_features = 4716, feature_vec_len = 1024, pad_batch_max = False):

		self.small2big = json.load(open('../Utils/small2big.json'))
		self.record_names = record_names
		self.reader = tf.TFRecordReader()
		self.sess = sess
		self.num_classifiers = num_classifiers
		self.num_features = num_features
		self.feature_vec_len = feature_vec_len
		self.pad_batch_max = pad_batch_max

		filename_queue = tf.train.string_input_producer(self.record_names)
		_, serialized_example = self.reader.read(filename_queue)

		self.contexts, self.features = tf.parse_single_sequence_example(
			serialized_example,
			context_features={
				'video_id': tf.FixedLenFeature([], tf.string),
				'labels': tf.VarLenFeature(tf.int64)
			},
			sequence_features = {
				'rgb': tf.FixedLenSequenceFeature([], dtype = tf.string)
			})
		self.frame_features = tf.reshape(tf.cast(tf.decode_raw(self.features['rgb'], tf.uint8), tf.float32), [-1, 1024])

	def fetch(self, batch_size):
		batch_data = []
		i = 0
		while i < batch_size:
			one_data = {}
			[frame_features, labels] = self.sess.run([self.frame_features, self.contexts['labels']])
			one_data['features'] = frame_features
			one_data['labels'] = labels[1]
			i += 1
			batch_data.append(one_data)
		return batch_data

	def preProcess(self, batch_data, classify, max_video_len  = 300):
		batch_size = len(batch_data)
		if self.pad_batch_max:
			max_video_len = max([d['features'].shape[0] for d in batch_data])
		labels_fine = None
		labels_rough = None
		if classify == 'SVM':
			labels_fine = np.negative(np.ones((batch_size, self.num_features)))
			labels_rough = np.negative(np.ones((batch_size, self.num_classifiers)))
		elif classify == 'lr':
			labels_fine = np.zeros((batch_size, self.num_features))
			labels_rough = np.zeros((batch_size, self.num_classifiers))
		else:
			print("Wrong parameter: classify. Input 'SVM' or 'lr'. B-Bye.")
			quit()
		pad_feature = np.empty((batch_size, max_video_len, self.feature_vec_len))
		original_len = np.zeros((0, 0))
		labels_rough = self.fine2rough(batch_data)
		i = 0
		for data in batch_data:
			#pdb.set_trace()
			labels_fine[i][data['labels']] = 1
			original_len = np.append(original_len, data['features'].shape[0])
			cut_length = max_video_len -  data['features'].shape[0]
			if cut_length>=0:
				tmp = np.append(data['features'], np.zeros((cut_length, self.feature_vec_len)), 0)
				pad_feature[i] = tmp
			else:
				pad_feature[i] = data['features'][:max_video_len, :]
			i += 1
		batch_data = {'labels_rough': labels_rough,
					  'labels_fine': labels_fine,
					  'pad_feature': pad_feature,
					  'original_len': original_len}
		return batch_data
	
	def fine2rough(self, batch_data):
		label_rough = np.zeros([len(batch_data), self.num_classifiers]) 
		for b, data in enumerate(batch_data):
			temp_rough = [self.small2big[str(s)] for s in data['labels']]
			rough_label = scp.mode(temp_rough)[0][0]
			label_rough[b, rough_label] = 1	
		return label_rough

def main():
	sess = tf.Session()
	tfr = tfReader(sess, ['../Data/train--.tfrecord', '../Data/train-0.tfrecord'])
	init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 
	sess.run(init)
	tf.train.start_queue_runners(sess = sess)
	data = tfr.fetch(1)
	prepared_data = tfr.preProcess(data, 'SVM')
	pdb.set_trace()

if __name__ == '__main__':
	main()
