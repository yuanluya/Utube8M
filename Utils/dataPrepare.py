import tensorflow as tf
import pdb
import numpy as np

class tfReader:
	def __init__(self, sess, record_names):
		
		self.record_names = record_names
		self.reader = tf.TFRecordReader()
		self.sess = sess

		filename_queue = tf.train.string_input_producer(self.record_names, num_epochs = 1)
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

	def preProcess(self, batch_data, need_better_name, num_features = 4716, feature_vec_len = 1024):
		batch_size = len(batch_data)
		max_len = max([d['features'].shape[0] for d in batch_data])
		labels = None
		if need_better_name == 'SVM':
			labels = np.negative(np.ones((batch_size, num_features)))
		elif need_better_name == 'lr':
			labels = np.zeros((batch_size, num_features))
		else:
			print("Wrong parameter: need_better_name. Input 'SVM' or 'lr'. B-Bye.")
			quit()
		pad_feature = np.empty((batch_size, max_len, feature_vec_len))
		original_len = np.zeros((0, 0))
		i = 0
		for data in batch_data:
			#pdb.set_trace()
			labels[i][data['labels']] = 1
			original_len = np.append(original_len, data['features'].shape[0])
			tmp = np.append(data['features'], np.zeros((max_len -  data['features'].shape[0], feature_vec_len)), 0)
			pad_feature[i] = tmp
			i += 1
		return labels, pad_feature, original_len


def main():
	sess = tf.Session()
	tfr = tfReader(sess, ['../Data/train--.tfrecord', '../Data/train-0.tfrecord'])
	init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 
	sess.run(init)
	tf.train.start_queue_runners(sess = sess)
	data = tfr.fetch(50)
	(A, B, C) = tfr.preProcess(data, 'SVM')
	C = C.astype(int)
	pdb.set_trace()

if __name__ == '__main__':
	main()
