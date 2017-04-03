import numpy as np
import tensorflow as tf
import os
import sys
import pdb
import time
import os.path as osp
from config import flags
from step import *


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))
NETWORK_DIR = osp.abspath(osp.join(ROOT_DIR, 'Networks'))
EXP_DIR = osp.abspath(osp.join(ROOT_DIR, 'Exp'))
#SUBMISSION_DIR = osp.abspath(osp.join(ROOT_DIR, 'Submission'))
#TRAIN_DIR = osp.abspath(osp.join(ROOT_DIR, 'TrainVal'))
UTIL_DIR = osp.abspath(osp.join(ROOT_DIR, 'Util'))
CHECKPOINTS_DIR = osp.abspath(osp.join(ROOT_DIR, 'Checkpoints'))
sys.path.append(ROOT_DIR)
sys.path.append(NETWORK_DIR)
sys.path.append(EXP_DIR)
#sys.path.append(SUBMISSION_DIR)
#sys.path.append(TRAIN_DIR)
sys.path.append(UTIL_DIR)
sys.path.append(CHECKPOINTS_DIR)


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
	if net.load(sess, '../Checkpoints', 'tcNet_%s' % flags.training_phase, restore_vars): # NAME!!
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
			avg_loss.append(step(sess, net, tfr, flags.batch_size, flags.loss_mode, flags.silent_step))

			current_iter += 1
			if current_iter % flags.snapshot_iter == 0:
				net.save(sess, '../Checkpoints', 'teNet_%s_%s' % (flags.training_phase ,str(current_iter)))
			#print('[$] iter timing: %d' % (time.clock() - t0))

if __name__ == '__main__':
	main()
