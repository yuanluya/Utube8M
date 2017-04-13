import numpy as np
import scipy.stats as scp
import sys
import pdb
import tensorflow as tf

sys.path.append('../Networks')
sys.path.append('../Utils')

from net import tcNet
from dataPrepare import tfReader

def step(sess, net, tfr, batch_size, mode, silent_step):
	data = tfr.fetch(batch_size)
	loss = None
	if mode == 'train':
		[_, loss_rough, loss_fine, cls_level1_prob, cls] = \
			sess.run([net.minimize, net.loss_rough, net.loss_fine, net.cls_level1_prob, net.cls],
			feed_dict = {net.frame_features: data['pad_feature'],
						 net.labels_fine: data['labels_fine'],
						 net.labels_rough: data['labels_rough'],
						 net.labels_fine_factor: data['labels_fine_factor'],
						 net.labels_rough_factor: data['labels_rough_factor'],
						 net.batch_lengths: data['original_len']})
	elif mode == 'val':
		[loss_rough, loss_fine, cls_level1_prob, cls] = \
			sess.run([net.loss_rough, net.loss_fine, net.cls_level1_prob, net.cls],
			feed_dict = {net.frame_features: data['pad_feature'],
						 net.labels_fine: data['labels_fine'],
						 net.labels_rough: data['labels_rough'],
						 net.labels_fine_factor: data['labels_fine_factor'],
						 net.labels_rough_factor: data['labels_rough_factor'],
						 net.batch_lengths: data['original_len']})
	elif mode == 'test':
		[cls] = sess.run([net.cls],
			feed_dict = {net.frame_features: data['pad_feature'],
						 net.batch_lengths: data['original_len']})
	pdb.set_trace()
	if not silent_step and mode != 'test':
		print('[ROUGH]', tfr.evaluator_rough.accumulate(cls_level1_prob, data['labels_rough'], loss_rough))
		print('[FINE] ', tfr.evaluator.accumulate(cls, data['labels_fine'], loss_fine), '\n')

if __name__ == '__main__':
	main()
