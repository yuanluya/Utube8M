import numpy as np
from easydict import EasyDict as edict
import os

flags = edict()
flags.data_dir = '../Data'

#model structures
flags.rnn_hidden_size = 1024 
flags.cls_feature_dim = 2048
flags.cnn_kernels = [[3, 1024, None, 1e-1],
 					 [3, 1024, 2, 1e-2],
					 [3, 512, 2, 1e-2],
					 [3, 256, 2, 1e-2]]

#model hyperparameters
flags.batch_size = 128
flags.learning_rate = 5e-5
flags.training_phase = 'phase1_lstm'

#training options
flags.mode = 'train' #<train|val|test>
flags.init_iter = 11000
flags.silent_train = False
flags.silent_step = False
flags.snapshot_iter = 500
flags.print_iter = 5
flags.max_iter= 5000
flags.model_init = 'None'

