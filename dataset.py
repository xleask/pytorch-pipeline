# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 21:10:35 2024

Generic dataset class for PyTorch model training.

@author: Scott
"""

from dataset_base import PyDatasetIO



class PyDataset(PyDatasetIO):
    '''
    Class for laoding and feeding data to model for training.
    Assumes all data can fit into memory.
    Assumes data can be put into a list-like structure, where the first dimension
    indexes the examples, and following dimensions represent input dimensions.
    '''
    def __init__(self, dataset_params):
        super(PyDataset, self).__init__(dataset_params)
        
    
    def __getitem__(self, index):
        '''
        Data are accessed via this method.
    
        index : integer or array
            Index/indices of input/arget data to return.    
        '''
        batch_sample = {}
        for data_key, data in self.data.items():
            sample = data['data']['image_data'][index]
            
            data_batch_transform = data['data_batch_transform']
            data_batch_transform_args = data['data_batch_transform_args']
            
            if data_batch_transform is not None:
                sample = data_batch_transform(sample, **data_batch_transform_args)
                
            batch_sample[data_key] = sample
            
        return batch_sample
    
    
def get_dataset(dataset_params):
    '''
    Return an instance of a PyDataset.
    
    dataset_args : dict
        {argument:value} arguments of PyDataset class.
    data_read_args : dict
        {argument:value} arguments for reading data from PyDataset
        read_data_streams method.
    '''
    dataset = PyDataset(dataset_params)
    return dataset

