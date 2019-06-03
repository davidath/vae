#! /usr/bin/env python

import yaml

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
