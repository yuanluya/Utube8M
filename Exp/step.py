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
		[_, loss, cls_level1_prob, cls_level1, rnn_features, cnn_output, cls_features_1] = \
			sess.run([net.minimize, net.loss, net.cls_level1_prob, net.cls_level1, net.rnn_features, net.cnn_output, net.cls_features_1],
			feed_dict = {net.frame_features: data['pad_feature'],
						 net.labels_fine: data['labels_fine'],
						 net.labels_rough: data['labels_rough'],
						 net.labels_rough_factor: data['labels_rough_factor'],
						 net.batch_lengths: data['original_len']})
	elif mode == 'val':
		[loss, cls_level1] = sess.run([net.loss, net.cls_level1],
			feed_dict = {net.frame_features: data['pad_feature'],
						 net.labels_fine: data['labels_fine'],
						 net.labels_rough: data['labels_rough'],
						 net.labels_rough_factor: data['labels_rough_factor'],
						 net.batch_lengths: data['original_len']})
	elif mode == 'test':
		[cls] = sess.run([net.cls],
			feed_dict = {net.frame_features: data['pad_feature'],
						 net.labels_rough_factor: data['labels_rough_factor'],
						 net.batch_lengths: data['original_len']})

	if not silent_step and mode != 'test':
		gt_labels = np.nonzero(data['labels_rough'])[1]
		first_argmax = np.argmax(cls_level1, 1)
		cls_level1[np.arange(128), first_argmax] = np.nan
		second_argmax = np.nanargmax(cls_level1, 1)
		count_gt = scp.mode(gt_labels)[1]
		count_pred = scp.mode(first_argmax)[1]
		top_accuracy = np.sum(gt_labels == first_argmax) / first_argmax.shape[0]
		top2_accuracy = np.sum(np.logical_or((gt_labels == first_argmax), 
					(gt_labels == second_argmax))) / first_argmax.shape[0]
		baseline = count_gt[0] / first_argmax.shape[0]
		print(first_argmax)
		print('accuracy: %f, top 2 accuracy: %f, baseline: %f, performance: %f' \
			% (top_accuracy, top2_accuracy, baseline, performance))
	return loss

if __name__ == '__main__':
	main()
