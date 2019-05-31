#! /usr/bin/env python

import tensorflow as tf
from tensorflow import keras as K

#############################################################################
# Recreating Variational Autoencoder from paper https://arxiv.org/abs/1312.6114
# "Auto-Encoding Variational Bayes"
#############################################################################

# Variational Autoencoder class containing topology and objective functions

class VAE:
    # Constructor / Build topology 
    # Args: Configuration dictionary loaded from yaml
    def __init__(self, conf):
        try:
        except KeyError:
            raise KeyError
        # End to end model
        self._e2e = K.models.Sequential()



