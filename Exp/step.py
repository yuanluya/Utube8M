import numpy
from config import *

def step(sess, net, labels, tfr, silent = True):
	
	# for idx, subbatch in enumerate(subbatches):
	# 	sub_shapes.append(subbatch['images'].shape[0])
	# 	[loss, seg_loss, _] = \
	# 		sess.run([net.loss, net.cross_entropy_mean, net.enqueue], \
	# 		  				  feed_dict = {net.im_input: np.array(subbatch['images']),
	# 		  			   	   			   net.seg_label: np.array(subbatch['labels'])})
	# 	total_loss.append(seg_loss)
	# #optimization
	# sess.run([net.train_op], feed_dict = {net.apply_grads_num: np.array(sub_shapes),
	# 										   net.batch_num: np.array([len(sub_shapes)])})
	# if not silent:
	# 	print('\t[!]segmentation loss: %f, total loss: %f' % (seg_loss, loss))
	# return np.mean(total_loss)
	data = tfr.fetch(batch_size)
	(labels, pad_feature, original_len) = tfr.preProcess(data, classifier)


	( _) = sess.run([net.func] \
			  				  feed_dict = {net.frame_features: np.array(pad_feature),
			  			   	   			   net.labels: np.array(labels)})
	if not silent:
		print('\t[!]loss: %f' % (net.loss))
