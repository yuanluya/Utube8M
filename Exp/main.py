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
				  flags.cls_feature_dim, flags.learning_rate,
				  flags.weight_decay, train = (flags.mode == 'train'))
		
	tfr = tfReader(sess, flags.data_dir, flags.mode, flags.rough_bias)
	init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 
	sess.run(init)
	tf.train.start_queue_runners(sess = sess)
	
	global_result_saver = {'good': [], 'bad': []}
	if flags.restore_mode == 'all':
		if net.load(sess, '../Checkpoints', 'tcNet_%s_%d' % (flags.init_model, flags.init_iter),
			net.variable_patches['rough_vars'] + net.variable_patches['fine_vars']):
			print('LOAD SUCESSFULLY')
		elif flags.mode == 'train':
			print('[!!!]No Model Found, Train From Scratch')
		else:
			print('[!!!]No Model Found, Cannot Test or Validate')
			return
	elif flags.restore_mode == 'seperate':
		if flags.mode != 'train':
			print('[!!!]No Model Found, Cannot Test or Validate, '
				'don\'t recommend train from scratch')
			return
		if net.load(sess, '../Checkpoints', 'tcNet_rough_%d' % flags.init_rough_iter, net.variable_patches['rough_vars']):
			print('LOAD ROUGH SUCESSFULLY')
		else:
			print('CANNOT LOAD ROUGH')
			assert(0)
		if net.load(sess, '../Checkpoints', 'tcNet_fine_%d' % flags.init_fine_iter, net.variable_patches['fine_vars']):
			print('LOAD ROUGH SUCESSFULLY')
		else:
			print('CANNOT LOAD FINE')
			assert(0)


	if flags.mode == 'train' or flags.mode == 'val':
		current_iter = 1
		while current_iter < flags.max_iter:
			t0 = time.clock()
			if current_iter % flags.print_iter == 0:
				result_fine = tfr.evaluator.get()
				result_rough = tfr.evaluator_rough.get()
				print('[RESULT]{iter %d, map: %f, gap: %f, avg_hit_@_one: %f, avg_perr %f}' %\
					(current_iter, np.sum(result_rough['aps']) / np.sum(np.array(result_rough['aps']) > 0),
					result_rough['gap'], result_rough['avg_hit_at_one'], result_rough['avg_perr']))
				print('{FINE, map: %f, gap: %f, avg_hit_@_one: %f, avg_perr %f}\n' %\
					(np.sum(result_fine['aps']) / np.sum(np.array(result_fine['aps']) > 0),
					result_fine['gap'], result_fine['avg_hit_at_one'], result_fine['avg_perr']))
			step(sess, net, tfr, flags.batch_size, flags.mode, flags.silent_step, global_result_saver)

			current_iter += 1
			if current_iter % flags.snapshot_iter == 0:
				net.save(sess, '../Checkpoints', 'tcNet_%s_%d' % (flags.save_model, flags.init_iter + current_iter))

if __name__ == '__main__':
	main()
