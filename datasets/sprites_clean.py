#! /usr/bin/python

import numpy as np
import sys

d = np.load(sys.argv[1])

samples = d['imgs'].shape[0]
features = d['imgs'].shape[1]

imgs = d['imgs'].reshape(samples, features**2)

np.savez_compressed('dsprites.npz', train_data=imgs[:samples-1], test_data=imgs[samples-1:samples], train_labels=np.zeros(shape=(1,1)), test_labels=np.zeros(shape=(1,1)))


