import numpy as np
from easydict import EasyDict as edict
import os

flags = edict()
flags.data_dir = '../Data'

#model structures
flags.rnn_hidden_size = 1024 
flags.cls_feature_dim = [4096, 2048, 1024, 1024]
flags.cnn_kernels = [[3, 1024, None, 1e-1],
 					 [3, 1024, 2, 1e-2],
					 [3, 512, 2, 1e-2],
					 [3, 256, 2, 1e-2]]

#model hyperparameters
flags.batch_size = 128 
flags.learning_rate = [1e-4, 0, 0]
flags.weight_decay = 1e-5
flags.rough_bias = np.array([1, 1, 5, 3, 3, 3, 3, 
							  1, 3, 3, 1, 5, 7, 3, 
					 		  5, 10, 5, 3, 1, 3,
					 		  3, 5, 1, 3, 3])

#training options
flags.mode = 'train' #<train|val|test>
flags.init_rough_iter = 22000   #used in seperate mode
flags.init_fine_iter = 57000	#used in seperate mode
flags.init_model = 'combine' 	#used in all mode
flags.save_model = 'combine' 
flags.init_iter = 0				#used in all mode
flags.silent_train = False
flags.silent_step = False
flags.snapshot_iter = 1000
flags.print_iter = 2
flags.max_iter = 10000
flags.restore_mode = 'seperate' #<all|seperate>

