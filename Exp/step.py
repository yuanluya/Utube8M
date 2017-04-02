import numpy
import sys
import pdb
import tensorflow as tf
from config import *

sys.path.append('../Networks')
sys.path.append('../Utils')

from net import tcNet
from dataPrepare import tfReader

def step(sess, net, tfr, silent = False):
	data = tfr.fetch(batch_size)
	(labels, pad_feature, original_len) = tfr.preProcess(data, classifier)
	[loss] = sess.run([net.loss], 
		feed_dict = {net.frame_features: pad_feature, net.labels: labels, net.batch_lengths: original_len})
	if not silent:
		print('\t[!]loss: %f' % (loss))

def main():
	device_idx = 0
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
					  log_device_placement = False))
	net = tcNet(sess)
	cnn_kernels = [[3, 2048, None],
				   [3, 4096, 2],
				   [3, 4096, None],
				   [3, 4096, 2],
				   [3, 2048, 2]]
	with tf.device('/gpu: %d' % device_idx): 
		net.build(2048, cnn_kernels, 4096, 1e-4)
	reader = tfReader(sess, ['../Data/train--.tfrecord', '../Data/train-0.tfrecord'])
	init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 
	sess.run(init)
	tf.train.start_queue_runners(sess = sess)

	step(sess, net, reader)

if __name__ == '__main__':
	main()
