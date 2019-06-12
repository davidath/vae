#! /usr/bin/env python

import yaml
import numpy as np
from dataset import Dataset
import os
#############################################################################
# General purpose functions
#############################################################################


# Logging messages such as loss,loading,etc.
# Args: string, label

def log(s, label='INFO'):
    sys.stdout.write(label + ' [' + str(datetime.now()) + '] ' + str(s) + '\n')
    sys.stdout.flush()

# Load yaml file configuration
# Args: Yaml configuration file path

def load_config(path):
    try:
        # Open file
        with open(path, 'r') as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError:
                log('YAML Syntax Error!', 'Exception')
    except FileNotFoundError:
        log('File '+ path + ' not found!', 'Exception')


# Get layer from array of nn configuration array with name reference
# Args: conf, layer name
# Returns: layer properties from yaml

def layer_by_name(conf, lname):
    _conf = conf['NNConf']
    return next(i for i in _conf['layers'] if i['name'] == 'Input')

# Checks npz keys


def check_keys(npz):
    #  keys = ['data', 'labels', 'test', 'tlab']
    keys = ['train_data', 'train_labels', 'test_data', 'test_labels']
    try:
        for k in keys:
            if k not in npz.keys():
                raise ValueError('npz structure missing key: ' + k + ', npz must' +
                                 'have keys [data, labels, test, tlab]')
    except:
        raise ValueError('npz structure missing key: ' + k + ', npz must' +
                         'have keys [data, labels, test, tlab]')


# Generic function for loading datasets


def load_data(path):
    if '.npz' not in path:
        raise TypeError("load_data currently handles only .npz structures")
    struct = np.load(path)
    check_keys(struct)
    try:
        # handle labels as integers
        #  trlab = struct['labels']
        trlab = struct['train_labels']
        trlab = trlab.reshape(trlab.shape[0], 1)
        #  telab = struct['tlab']
        telab = struct['test_labels']
        telab = telab.reshape(telab.shape[0], 1)
        #  return Dataset(struct['data'], struct['test'], trlab, telab)
        return Dataset(struct['train_data'], struct['test_data'], trlab, telab)
    except:
        #  trlab = struct['labels']
        trlab = struct['train_labels']
        #  telab = struct['tlab']
        telab = struct['test_labels']
        #  d = Dataset(struct['data'], struct['test'], trlab, telab)
        d = Dataset(struct['train_data'], struct['test_data'], trlab, telab)
        d.set_train_ones(trlab)
        d.set_test_ones(telab)
    return d


# Access full dataset permutation

def get_perm(perm_str, XX):
    if os.path.exists(perm_str):
        p = np.load(perm_str)
    else:
        p = np.random.permutation(XX.shape[0])
        np.save(perm_str, p)
    return p


