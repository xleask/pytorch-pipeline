# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 21:11:40 2024

Example dataset input/target transforms and functions/readers for dataset.py
module.

@author: Scott
"""

import h5py
import numpy as np
import torch


def example_data_reader(path):
    '''
    Example h5 data reader.
    '''
    hf = h5py.File(path, 'r')
    image_data = hf['data'][:]
    image_data = np.expand_dims(image_data, axis=1)
    
    # Normalize data
    image_data = torch.tensor(np.float32(image_data)/255)

    n_samples = len(image_data)
    all_data = {
        'image_data': image_data,
        'n_inputs': n_samples,
        }
    return all_data


def example_batch_transform(image_inputs, augment=True, add_image_noise=True):
    '''
    Example transform to apply to training batch.
    '''
    if not augment:
        return image_inputs
    
    if add_image_noise:
        image_noise = torch.normal(0, 0.01, size=image_inputs.shape)
        image_noise = image_noise.to(image_inputs.device)
        image_inputs = image_inputs + image_noise
        
    return image_inputs


def example_data_postprocessor(data):
    '''
    Example postprocessor once all data files have been read.
    '''
    return data[0]