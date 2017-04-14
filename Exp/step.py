import numpy as np
import scipy.stats as scp
import sys
import pdb
import tensorflow as tf

sys.path.append('../Networks')
sys.path.append('../Utils')

from net import tcNet
from dataPrepare import tfReader

def step(sess, net, tfr, batch_size, mode, silent_step, result_saver):
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
	rough_result = tfr.evaluator_rough.accumulate(cls_level1_prob, data['labels_rough'], loss_rough)
	fine_result = tfr.evaluator.accumulate(cls, data['labels_fine'], loss_fine)
	find_result(result_saver, cls, data['labels_fine'], data['video_ids'])
	if not silent_step and mode != 'test':
		print('[ROUGH]', rough_result)
		print('[FINE] ', fine_result, '\n')

#result saver with field: good|bad
def find_result(result_saver, pred_probs, labels, v_ids):
	cls_pred = np.argsort(pred_probs, axis = 1)
	for i in range(pred_probs.shape[0]):
		gt_labels = np.nonzero(labels[i, :])
		num_labels = gt_labels.shape[0]
		top_k = cls_pred[i, 0: num_labels].sort()
		if np.sum(gt_labels == top_k) == num_labels:
			result_saver['good'].append(v_ids[i])
		elif np.sum(gt_labels == top_k) == 0:
			result_saver['bad'].append(v_ids[i])

if __name__ == '__main__':
	main()
