import numpy as np
#from easydict import EasyDict as edict

model_init = -1
model_save =  -1
snapshot_iter = 200
learning_rate = 1e-4
batch_size = 50
train_mode = True
classifier = 'SVM'
silent_train = True

max_iter = 10


rnn_hidden_size = 2048
cls_feature_dim = 4096
cnn_kernels = [[3, 2048, None],
				   [3, 4096, 2],
				   [3, 4096, None],
				   [3, 4096, 2],
				   [3, 2048, 2]]

