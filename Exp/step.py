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
		[_, loss, cls_level1_prob, cls_level1, cls] = \
			sess.run([net.minimize, net.loss, net.cls_level1_prob, net.cls_level1, net.cls],
			feed_dict = {net.frame_features: data['pad_feature'],
						 net.labels_fine: data['labels_fine'],
						 net.labels_rough: data['labels_rough'],
						 net.labels_fine_factor: data['labels_fine_factor'],
						 net.labels_rough_factor: data['labels_rough_factor'],
						 net.batch_lengths: data['original_len']})
	elif mode == 'val':
		[loss, cls_level1, cls] = sess.run([net.loss, net.cls_level1, net.cls],
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
	if not silent_step and mode != 'test':
		print('[1]accuracy: %f, top 2 accuracy: %f, baseline: %f, performance: %f, unique: %d/%d' \
			% (top_accuracy, top2_accuracy, baseline, performance, num_unique_gt, num_unique_pred))
		print('\t[2]', tfr.evaluator.accumulate(cls, data['labels_fine'], loss))
	return loss

if __name__ == '__main__':
	main()
