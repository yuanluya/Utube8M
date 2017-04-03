import numpy
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
	[_, loss] = sess.run([net.minimize, net.loss],
		feed_dict = {net.frame_features: data['pad_feature'],
					 net.labels_fine: data['labels_fine'],
					 net.labels_rough: data['labels_rough'],
					 net.batch_lengths: data['original_len']})
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
