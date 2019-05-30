#! /usr/bin/env python

import tensorflow as tf
import os
import sys
import yaml

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

