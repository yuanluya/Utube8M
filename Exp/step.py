import numpy as np
import sys
import pdb
import tensorflow as tf

sys.path.append('../Networks')
sys.path.append('../Utils')

from net import tcNet
from dataPrepare import tfReader

def step(sess, net, tfr, batch_size, loss_mode, silent_step):
	data = tfr.fetch(batch_size)
	data = tfr.preProcess(data, loss_mode)
	[_, loss, rnn_features, cnn_output, cls_features, cls_level1] = sess.run([net.minimize, net.loss, net.rnn_features, net.cnn_output, net.cls_features, net.cls_level1],
		feed_dict = {net.frame_features: data['pad_feature'],
					 net.labels_fine: data['labels_fine'],
					 net.labels_rough: data['labels_rough'],
					 net.labels_rough_factor: data['labels_rough_factor'],
					 net.batch_lengths: data['original_len']})
	gt_labels = np.nonzero(data['labels_rough'])[1]
	prediction = np.argmax(cls_level1, 1)
	print('accuracy: %f'% (np.sum(gt_labels == prediction) / prediction.shape[0]))
	print(prediction)
	#pdb.set_trace()
	if not silent_step:
		print('\t[!]loss: %f' % (loss))
	return loss

def main():
	device_idx = 0
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
					  log_device_placement = False))
	net = tcNet(sess)
	with tf.device('/gpu: %d' % device_idx): 
		net.build(rnn_hidden_size, cnn_kernels, cls_feature_dim, 'phase1', learning_rate)
	reader = tfReader(sess, ['../Data/train--.tfrecord', '../Data/train-0.tfrecord'])
	init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 
	sess.run(init)
	tf.train.start_queue_runners(sess = sess)

	step(sess, net, reader)

if __name__ == '__main__':
	main()
