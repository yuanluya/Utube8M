import numpy as np
from easydict import EasyDict as edict

flags = edict()
flags.data_files = ['../Data/train--.tfrecord']\
	+ ['../Data/train-%d.tfrecord' % s for s in range(10)]

#model structures
flags.rnn_hidden_size = 2048
flags.cls_feature_dim = 4096
flags.cnn_kernels = [[3, 2048, None],
 					 [3, 4096, 2],
					 [3, 4096, None],
					 [3, 4096, 2],
					 [3, 2048, 2]]

#model hyperparameters
flags.batch_size = 32
flags.learning_rate = 1e-3
flags.training_phase = 'phase1'

#training options
flags.train_mode = True
flags.silent_train = False
flags.silent_step = True
flags.loss_mode = 'lr'
flags.snapshot_iter = 500
flags.print_iter = 20
flags.max_iter= 3000
flags.model_init = 'None'

