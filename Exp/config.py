import numpy as np
from easydict import EasyDict as edict
import os

flags = edict()
flags.data_dir = '../Data'

#model structures
flags.rnn_hidden_size = 1024 
flags.cls_feature_dim = [4096, 2048, 1024]
flags.cnn_kernels = [[3, 1024, None, 1e-1],
 					 [3, 1024, 2, 1e-2],
					 [3, 512, 2, 1e-2],
					 [3, 256, 2, 1e-2]]

#model hyperparameters
flags.batch_size = 50
flags.learning_rate = [0, 1e-4, 1e-5]
flags.weight_decay = 2e-4
flags.training_phase = 'phase2' #<phase1|phase2|phasae3>
flags.init_model = 'phase1_lstm_correct'
flags.save_model = 'phase2'

#training options
flags.mode = 'train' #<train|val|test>
flags.init_iter = 27600
flags.silent_train = False
flags.silent_step = False
flags.snapshot_iter = 400
flags.print_iter = 5
flags.max_iter= 10000
flags.restore_mode = 'old' #<all|old>

