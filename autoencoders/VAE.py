#! /usr/bin/env python

import tensorflow as tf
from tensorflow import keras as K

# Get layer from array of nn configuration array with name reference
# Args: conf, layer name
# Returns: layer properties from yaml

def layer_by_name(layers, lname):
    return next(i for i in layers if i['name'] == lname)

# Perform reparameterazation trick
# Args: mu and sigma tensors

def reparam_trick(args):
    mu, sigma = args
    # Dynamic sample & feature
    samples = K.backend.shape(mu)[0]
    features = K.backend.shape(mu)[1]
    
    epsilon = K.backend.random_normal(shape=(samples, features))
    # z = μ + σ^2 * ε
    return mu + K.backend.exp(0.5 * sigma) * epsilon


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
            #  Building latent space
            _mn_layer = layer_by_name(_layers, 'MultiNormalParams')
            # Layer Weight initialization
            W_INIT = tf.truncated_normal_initializer(mean=0.0, stddev=0.01,
                                            seed=conf['seed'])
            # Encoder
            self._in, self._z, self._encoder = self.encoder(conf, W_INIT)
            # Decoder
            _z_in = K.layers.Input(shape=(_mn_layer['properties'][0]['Width'], ),
                    name='z_in')
            out, self._decoder = self.decoder(conf, _z_in, W_INIT)
            # VAE
            self._out = self._decoder(self._encoder(self._in))
            self._ae = K.Model(self._in, self._out)
            # Generator
            self._gen_in = K.layers.Input(shape=(_mn_layer['properties'][0]['Width'], ),
                                name='gen_in')
            self._gen_out, self._gen = self.decoder(conf, self._gen_in, W_INIT)
            self.loss_func()
        except KeyError:
            raise KeyError

    # Build encoder
    # Args: Configuration dictionary loaded from yaml, Weight initialization
    # function

    def encoder(self, conf, W_INIT):
        # Access nn configuration dictionary
        _conf = conf['NNConf']
        _layers = _conf['layers']
        # Check input channels
        _in = layer_by_name(_layers, 'Input')
        if _in['properties'][0]['Channels'] == 1:
            curr_width = _in['properties'][0]['Width']
            # Save input layer reference
            _in = K.layers.Input(shape=(curr_width, ))
        else:
            # Multi-channel (TODO)
            pass

        # Stacking hidden layers
        encoder = layer_by_name(_layers, 'Encoder')
        for c, i in enumerate(encoder['properties']):
            # First hidden is connected to input
            if c == 0:
                hidden = K.layers.Dense(units=i['Width'],
                                        activation=i['Activation'],
                                        kernel_initializer=W_INIT,
                                        name='Enc'+str(c))(_in)
            else:
                hidden = K.layers.Dense(units=i['Width'],
                                        activation=i['Activation'],
                                        kernel_initializer=W_INIT,
                                        name='Enc'+str(c))(hidden)
        #  Building latent space
        _mn_layer = layer_by_name(_layers, 'MultiNormalParams')
        # MU properties
        _mu_l = _mn_layer['properties'][0]
        # log SIGMA properties
        _sigma_l = _mn_layer['properties'][1]
        # Mu Layer
        mu = K.layers.Dense(units=_mu_l['Width'],
                                    activation=_mu_l['Activation'],
                                    kernel_initializer=W_INIT,
                                    name='mu')(hidden)
        # Log sigma layer
        log_sigma = K.layers.Dense(units=_sigma_l['Width'],
                                    activation=_sigma_l['Activation'],
                                    kernel_initializer=W_INIT,
                                    name='sigma')(hidden)
        # Latent representations
        z = K.layers.Lambda(reparam_trick,
                output_shape=(_mu_l['Width'],), name='Z')([mu, log_sigma])
        # Input -> Z
        _z = K.Model(_in, z, name='Encoder')
        return _in, z, _z

    # Build decoder
    # Args: Configuration dictionary loaded from yaml, latent input, 
    # Weight initialization function

    def decoder(self, conf, z, W_INIT):
        # Access nn configuration dictionary
        _conf = conf['NNConf']
        _layers = _conf['layers']
        # Stacking decoder layers
        decoder = layer_by_name(_layers, 'Decoder')
        for c, i in enumerate(decoder['properties']):
            # First decoding layer is connected to Z
            if c == 0:
                dec = K.layers.Dense(units=i['Width'],
                                    activation=i['Activation'],
                                    kernel_initializer=W_INIT,
                                    name='Dec'+str(c))(z)
            else:
                dec = K.layers.Dense(units=i['Width'],
                                    activation=i['Activation'],
                                    kernel_initializer=W_INIT,
                                    name='Dec'+str(c))(dec)
        # Check output channels
        _out = layer_by_name(_layers, 'Output')
        if _out['properties'][0]['Channels'] == 1:
            curr_width = _out['properties'][0]['Width']
            act = _out['properties'][0]['Activation']
            out = K.layers.Dense(units=curr_width,
                                    activation=act,
                                    kernel_initializer=W_INIT,
                                    name='Out')(dec)
        else:
            # Multi-channel (TODO)
            pass
        # Z -> X_recon
        _dec = K.Model(z, out, name='Decoder')
        return out, _dec

    # Build loss
    # Args: 

    def loss_func(self):
        # mse
        recon_loss = K.losses.mean_squared_error(self._in, self._out)
        
        # Get mu and sigma layers
        mu = next(layer.output for layer in self._encoder.layers if layer.name == 'mu')
        sigma = next(layer.output for layer in self._encoder.layers if layer.name == 'sigma')

        # Kl_div 
        kl_loss = 1 + sigma - K.backend.square(mu) - K.backend.exp(sigma)
        kl_loss = -0.5 * K.backend.sum(kl_loss, axis=-1)

        vae_loss = K.backend.mean(recon_loss + (1.0 * kl_loss))

        # Optimizing
        self._ae.add_loss(vae_loss)
        self._ae.compile(optimizer='adam')



