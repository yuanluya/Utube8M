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
	
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
											log_device_placement = False))
	
	net = tcNet(sess)
	with tf.device('/%s: %d' % (device, device_idx)): 
		net.build(flags.rnn_hidden_size, flags.cnn_kernels,
				  flags.cls_feature_dim, flags.training_phase, 
				  flags.learning_rate, flags.weight_decay, train = (flags.mode == 'train'))
		
	tfr = tfReader(sess, flags.data_dir, flags.mode, flags.rough_bias)
	init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 
	sess.run(init)
	tf.train.start_queue_runners(sess = sess)
	
	if flags.restore_mode == 'all':
		restore_vars = []
	elif flags.restore_mode == 'old':
		restore_vars = net.varlist 
	if net.load(sess, '../Checkpoints', 'tcNet_%s_%d' % (flags.init_model, flags.init_iter), restore_vars):
		print('LOAD SUCESSFULLY')
	elif flags.mode == 'train':
		print('[!!!]No Model Found, Train From Scratch')
	else:
		print('[!!!]No Model Found, Cannot Test or Validate')
		return

	if flags.mode == 'train' or flags.mode == 'val':
		current_iter = 1
		avg_loss = []
		while current_iter < flags.max_iter:
			t0 = time.clock()
			if current_iter % flags.print_iter == 0:
				print('{iter %d}' % (current_iter))
				result = tfr.evaluator_rough.get()
				print('[RESULT]{iter %d, gap: %f, avg_hit_@_one: %f, avg_perr %f}' %\
					(current_iter, result['gap'], result['avg_hit_at_one'], result['avg_perr']))
				avg_loss = []
			loss = step(sess, net, tfr, flags.batch_size, flags.mode, flags.silent_step)
			avg_loss.append(loss)

			current_iter += 1
			if current_iter % flags.snapshot_iter == 0:
				net.save(sess, '../Checkpoints', 'tcNet_%s_%d' % (flags.save_model, flags.init_iter + current_iter))

if __name__ == '__main__':
	main()
