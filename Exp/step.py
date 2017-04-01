import numpy

def step(sess, net, data_loader, cv_empty, cv_full, silent = True):
	
	#access shared data
	cv_empty.acquire()
	while len(data_loader) == 0:
		cv_empty.wait()
	subbatches = data_loader.pop(0)
	cv_full.notify()
	cv_empty.release()
	total_loss = []
	sub_shapes = []
	for idx, subbatch in enumerate(subbatches):
		sub_shapes.append(subbatch['images'].shape[0])
		[loss, seg_loss, _] = \
			sess.run([net.loss, net.cross_entropy_mean, net.enqueue], \
			  				  feed_dict = {net.im_input: np.array(subbatch['images']),
			  			   	   			   net.seg_label: np.array(subbatch['labels'])})
		total_loss.append(seg_loss)
	#optimization
	sess.run([net.train_op], feed_dict = {net.apply_grads_num: np.array(sub_shapes),
											   net.batch_num: np.array([len(sub_shapes)])})
	if not silent:
		print('\t[!]segmentation loss: %f, total loss: %f' % (seg_loss, loss))
	return np.mean(total_loss)