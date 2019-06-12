#! /bin/bash

wget -nc https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz

python sprites_clean.py dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz

