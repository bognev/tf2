import tensorflow as tf
import h5py
import tensorflow_io as tfio
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random

print(tf.__version__, tfio.__version__)

batch_size = 256;

class generator:
    def __init__(self, file):
        self.file = file

    def __call__(self, *args, **kwargs):
        with h5py.File(self.file, 'r') as f:
            datasize = len(f['X'])
            idxs = list(range(0,datasize//batch_size))
            random.shuffle(idxs)
            for i in idxs:
                yield f['X'][range(batch_size)]
            f.close()

hdf5_path = '/home/bognev/Downloads/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5'
dataset = tf.data.Dataset.from_generator(generator(hdf5_path), tf.float32, tf.TensorShape([batch_size, 1024, 2])) #2555904, 1024, 2

for i, ex in enumerate(dataset):
    print(i, ex.shape)


