#! /usr/bin/env python

import tensorflow as tf
from tensorflow import keras as K

# Get layer from array of nn configuration array with name reference
# Args: conf, layer name
# Returns: layer properties from yaml

def layer_by_name(layers, lname):
    return next(i for i in layers if i['name'] == lname)

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
            # Access nn configuration dictionary
            _conf = conf['NNConf']
            _layers = _conf['layers']

            # Layer Weight initialization
            W_INIT = tf.truncated_normal_initializer(mean=0.0, stddev=0.01,
                                            seed=conf['seed'])

            # Check input channels
            _in = layer_by_name(_layers, 'Input')
            if _in['properties'][0]['Channels'] == 1:
                curr_width = _in['properties'][0]['Width']
                # Save input layer reference
                self._in = tf.keras.layers.Input(shape=(curr_width, ))
            else:
                # Multi-channel (TODO)
                pass

            # Stacking hidden layers
            encoder = layer_by_name(_layers, 'Encoder')
            for c, i in enumerate(encoder['properties']):
                # First hidden is connected to input
                if c == 0:
                    hidden = tf.keras.layers.Dense(units=i['Width'],
                                            activation=i['Activation'],
                                            kernel_initializer=W_INIT)(self._in)
                else:
                    hidden = tf.keras.layers.Dense(units=i['Width'],
                                            activation=i['Activation'],
                                            kernel_initializer=W_INIT)(hidden)
            #  Building latent space
            _mn_layer = layer_by_name(_layers, 'MultiNormalParams')
            # MU properties
            _mu_l = _mn_layer['properties'][0]
            # log SIGMA properties
            _sigma_l = _mn_layer['properties'][1]
            # Mu Layer
            mu = tf.keras.layers.Dense(units=_mu_l['Width'],
                                        activation=_mu_l['Activation'],
                                        kernel_initializer=W_INIT)(hidden)
            # Log sigma layer
            log_sigma = tf.keras.layers.Dense(units=_sigma_l['Width'],
                                        activation=_sigma_l['Activation'],
                                        kernel_initializer=W_INIT)(hidden)

            self._z = tf.keras.Model(self._in, [mu, log_sigma], name='Encoder')
            self._z.summary()
        except KeyError:
            raise KeyError



