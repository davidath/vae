#! /usr/bin/env python

import tensorflow as tf
import os
import sys
import yaml
import utils

sys.path.append('autoencoders')
from VAE import *

##############################################################################
# Training script for all VAE variations and datasets
##############################################################################

# Suppress tensorflow warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# Session settings to avoid pre allocating all the GPU memory

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

# Load configuration from yaml
config = utils.load_config(sys.argv[1])

# Assert "deterministic" behaviour for the experiments
tf.set_random_seed(config['seed'])

vae = VAE(config)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

vae._ae.fit(x_train.reshape(60000, 784), epochs=100, batch_size=256)
