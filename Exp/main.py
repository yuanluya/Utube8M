import numpy as np
import tensorflow as tf
import os
import sys
import pdb
import time
import os.path as osp
from config import flags
from step import *

def main():
	#set device
	if len(sys.argv) != 3 or \
		(sys.argv[1] != 'cpu' and sys.argv[1] != 'gpu'):
		print('wrong arguments, <cpu|gpu> <device_idx(integer)> ')
		return
	device = sys.argv[1]
	device_idx = int(sys.argv[2])

	if flags.train_mode:
		split = 'train'
	else:
		split = 'val'
	
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
											log_device_placement = False))
	
	net = tcNet(sess)
	with tf.device('/%s: %d' % (device, device_idx)): 
		net.build(flags.rnn_hidden_size, flags.cnn_kernels,
			flags.cls_feature_dim, flags.training_phase, flags.learning_rate)
		
	tfr = tfReader(sess, flags.data_files)
	init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 
	sess.run(init)
	tf.train.start_queue_runners(sess = sess)
	# if model_init == model_save:
	restore_vars = []
	# else:
	# 	restore_vars = net.varlist
	if net.load(sess, '../Checkpoints', 'tcNet_%s_%d' % (flags.training_phase, flags.init_iter), restore_vars): # NAME!!
		print('LOAD SUCESSFULLY')
	else:
		print('[!!!]No Model Found, Train From Scratch')
	# else:
	# 	print('[!!!]No Model Found, Cannot Test')
	# 	return

	if flags.train_mode:
		# silent_train = True
		# classifier = 'SVM' #['SVM', 'lr']
		current_iter = 1
		avg_loss = []
		while current_iter < flags.max_iter:
			t0 = time.clock()
			#print('{iter %d}' % (current_iter))
			if current_iter % flags.print_iter == 0:
				print('{iter %d}' % (current_iter))
				print('[#]average seg loss is: %f' % np.mean(avg_loss))
				avg_loss = []
			loss = step(sess, net, tfr, flags.batch_size, flags.loss_mode, flags.silent_step)
			avg_loss.append(loss)

			current_iter += 1
			if current_iter % flags.snapshot_iter == 0:
				net.save(sess, '../Checkpoints', 'tcNet_%s_%s' % (flags.training_phase ,str(flags.init_iter + current_iter)))
			#print('[$] iter timing: %d' % (time.clock() - t0))

if __name__ == '__main__':
	main()
