import tensorflow as tf
import pdb

reader = tf.TFRecordReader()
filename = "train--.tfrecord"
filename_queue = tf.train.string_input_producer([filename], num_epochs = 1)
_, serialized_example = reader.read(filename_queue)

contexts, features = tf.parse_single_sequence_example(
	serialized_example,
	context_features={
		'video_id': tf.FixedLenFeature([], tf.string),
		'labels': tf.VarLenFeature(tf.int64)
	},
	sequence_features = {
		feature_name: tf.FixedLenSequenceFeature([], dtype = tf.string)
		for feature_name in ['rgb', 'audio']
	})
features['rgb'] = tf.reshape(tf.cast(tf.decode_raw(features['rgb'], tf.uint8), tf.float32), [-1, 1024])
features['audio'] = tf.reshape(tf.cast(tf.decode_raw(features['audio'], tf.uint8), tf.float32), [-1, 128])
sess = tf.Session()
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 
sess.run(init)
tf.train.start_queue_runners(sess=sess)
i = 0
while i > -1:
	[features_1, features_2, labels] = sess.run(features.values() + [contexts['labels']])
	i += 1
	print(i, labels)
	print(features_1.shape)
