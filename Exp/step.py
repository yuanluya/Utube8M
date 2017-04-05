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
		[_, loss, cls_level1] = sess.run([net.minimize, net.loss, net.cls_level1],
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
		prediction = np.argmax(cls_level1, 1)
		count = scp.mode(prediction)[1]
		accuracy = np.sum(gt_labels == prediction) / prediction.shape[0]
		baseline = count[0] / prediction.shape[0]
		print('accuracy: %f, baseline: %f' % (accuracy, baseline))
	return loss

if __name__ == '__main__':
	main()
