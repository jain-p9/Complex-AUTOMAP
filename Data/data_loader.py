import numpy as np
import tensorflow as tf
import os

import config

train_path = config.train_path   #directory with .npy train files
test_path = config.test_path 

mask = np.load(config.mask_path)
mask = mask + 1j*mask

def read_npy_file(np_data_path, item):
    cropped_im = np.load(np_data_path.numpy().decode()+item.numpy().decode())
    im = 4096*cropped_im
    kspace = np.fft.fftshift(np.fft.fft2(im))
    kspace = ( 0.0075/(2*4096) )*kspace
    masked_kspace = np.squeeze(kspace*mask)
    
    mag = np.abs(masked_kspace)
    phase = np.angle(masked_kspace)
    constant = 0.0+1.0j
    mag = (mag-np.min(mag))/(np.max(mag)-np.min(mag))
    masked_kspace = np.multiply(mag.astype(np.complex64),np.exp(constant*phase.astype(np.complex64)))
    
    x_real = np.real(masked_kspace)
    x_imag = np.imag(masked_kspace)
    x = np.stack([x_real,x_imag], axis=-1)
    
    gr = np.abs(cropped_im)
    gr = (gr-np.min(gr))/(np.max(gr)-np.min(gr))
    
    return x, np.expand_dims(gr, axis=-1)

filenames = os.listdir(train_path)
train_dataset = tf.data.Dataset.from_tensor_slices(filenames)
train_dataset = train_dataset.map(
        lambda item: tuple(tf.py_function(read_npy_file, [train_path, item], [tf.float32, tf.float32])))


filenames = os.listdir(test_path)
test_dataset = tf.data.Dataset.from_tensor_slices(filenames)
test_dataset = test_dataset.map(
        lambda item: tuple(tf.py_function(read_npy_file, [test_path, item], [tf.float32, tf.float32])))


batch_size = config.batch_size
test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
