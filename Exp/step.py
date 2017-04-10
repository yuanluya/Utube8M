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
		gt_labels = np.nonzero(data['labels_rough'])[1]
		ranking = np.argsort(cls_level1, axis = 1)
		first_argmax = ranking[:, -1]
		second_argmax = ranking[:, -2]
		count_gt = scp.mode(gt_labels)[1]
		count_pred = scp.mode(first_argmax)[1]
		num_unique_pred = np.unique(first_argmax).shape[0]
		num_unique_gt = np.unique(gt_labels).shape[0]
		top_accuracy = np.sum(gt_labels == first_argmax) / first_argmax.shape[0]
		top2_accuracy = np.sum(np.logical_or((gt_labels == first_argmax), 
					(gt_labels == second_argmax))) / first_argmax.shape[0]
		baseline = count_gt[0] / first_argmax.shape[0]
		performance = count_pred[0] / first_argmax.shape[0]
		print(first_argmax)
		print('[1]accuracy: %f, top 2 accuracy: %f, baseline: %f, performance: %f, unique: %d/%d' \
			% (top_accuracy, top2_accuracy, baseline, performance, num_unique_gt, num_unique_pred))
		if net.phase != 'phase1':
			pos = cls > 0.1
			print(cls)
			print('\t[2]gt_labels: %d' % np.nonzero(data['labels_fine'])[0].shape[0])
			print('\t[2]predictions: %d' % np.sum(pos))
			print('\t[2]gt_diversity: %d' % np.unique(np.nonzero(data['labels_fine'])[1]).shape[0])
			print('\t[2]pred_diversity: %d' % np.unique(np.nonzero(pos)[1]).shape[0])
			print('\t[2]', tfr.evaluator.accumulate(cls, data['labels_fine'], loss))
	return loss

if __name__ == '__main__':
	main()
