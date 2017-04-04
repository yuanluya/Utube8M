import numpy as np
from easydict import EasyDict as edict
import os

flags = edict()
flags.data_files = ['../Data/%s' % f for f in os.listdir('../Data') if f[-8: ] == 'tfrecord']

#model structures
flags.rnn_hidden_size = 1024 
flags.cls_feature_dim = 2048
flags.cnn_kernels = \
					[[3, 1024, None, 1e-1],
 					 [3, 1024, 2, 1e-1],
					 [3, 512, 2, 1e-2],
					 [3, 256, 2, 1e-2]]

#model hyperparameters
flags.batch_size = 128
flags.learning_rate = 1e-4
flags.training_phase = 'phase1'

#training options
flags.train_mode = True
flags.init_iter = 0
flags.silent_train = False
flags.silent_step = True
flags.loss_mode = 'lr'
flags.snapshot_iter = 500
flags.print_iter = 5
flags.max_iter= 3000
flags.model_init = 'None'

