#! /usr/bin/env python

import tensorflow as tf
import os
import sys
import yaml
import utils
import numpy as np

sys.path.append('autoencoders')
from VAE import *
from decay_scheduler import *

##############################################################################
# Training script for all VAE variations and datasets
##############################################################################

# Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# Suppress sklearn warnings

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Session settings to avoid pre allocating all the GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

# Load configuration from yaml
config = utils.load_config(sys.argv[1])

# Assert "deterministic" behaviour for the experiments
tf.set_random_seed(config['seed'])

# Init NN
vae = VAE(config)

# Create save directory if it doesn't exist (Primary AE)
directory = config['model_output_path']
if not os.path.exists(directory):
    os.makedirs(directory)

# Output strings
out = config['model_output_path']
perm_str = out + config['prefix'] + '/' + config['enumber'] + '_perm.npy'

# Load dataset
if config['data_input_path'] is None:
    # Load MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Gather all samples
    X = np.concatenate(x_train, x_test)
    Y = np.concatenate(y_train, y_test)
else:
    dataset = utils.load_data(config['data_input_path'])
    try:
        (X_train, Y_train, X_test, Y_test) = (dataset.train.images, dataset.train.labels, 
                                           dataset.test.images,
                                           dataset.test.labels)
    except:
        X_train, X_test = dataset.train.images, dataset.test.images

# Gather all samples
X = np.concatenate((X_train, X_test))
Y = np.concatenate((Y_train, Y_test))

# Save/Load random permutation
p = utils.get_perm(perm_str, X)
# Randomize order of samples
X = X[p]
try:
    Y = Y[p]
except:
    pass

print(X.shape)
# Prepare for training (starting lr, decay amount)
decay_scheduler = decay_lr(
           config['NNConf']['hyperparameters']['lr_decay_epoch'], 
           0.1)
try:
    # Train
    vae._ae.fit(X,
                  epochs=config['NNConf']['hyperparameters']['max_epochs'], 
                  batch_size=config['NNConf']['hyperparameters']['batch_size'])
except KeyboardInterrupt:
    pass
