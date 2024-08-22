# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 21:11:32 2024

Base functionality for PyTorch datasets.

@author: Scott
"""

import os

import h5py

from torch.utils.data import Dataset

import dataset_functions



class PyDatasetIO(Dataset):
    '''
    Base class for PyTorch dataset, giving data reading functionality.
    
    data_keys : tuple of strings
        Key/name of each data stream to read, for easy referencing/model flow.
    data_len_key : string
        Specific key from data_keys to use for len(dataset).
    n_inputs : integer
        Number of inputs to read from given dataset option. If -1, all inputs
        are read.
    data_batch_transform : function or None
        If provided, applies a function to each data batch before passing through model.
    randomize_inputs : boolean
        If n_inputs < len(dataset), choose whether to take the first n_inputs
        (randomize_inputs=False) or a random selection (True).
    '''
    def __init__(self, dataset_params, **_):
        self.SUPPORTED_DATA_READ_TYPES = ('folder', 'h5', 'file', 'function')
        
        self.dataset_params = dataset_params
        
        self.data_len_key = dataset_params['data_len_key']
        self.data_keys = tuple([key for key in dataset_params.keys() if key != 'data_len_key'])
        
        self.data = {data_key: {} for data_key in self.data_keys}
        
        self.read_data_streams(dataset_params)
        
        
    def __len__(self):
        return self.data[self.data_len_key]['data']['n_inputs']
    
    
    def read_data_streams(self, dataset_params, **_):
        '''
        data_type : string
            One of ('input', 'target')
        data_read_type : string
            Dataset is created/read from specific type, one of:
                ('folder', 'h5', 'file', 'function')
            'folder' type: Read data from all files in given folder.
            'h5' type: Read data from h5 file.
            'file' type: Read data from specific file type. Requires file_Reader.
            'function' type: Generate data from given function.
        data_path : string or None
            For 'folder' data_read_type, directory must be provided. For 'h5' and
            'csv' data_read_type, specific file(s) must be given.
        file_reader : function or None
            For 'folder' and 'file' data_read_type, a function mst be provided
            to read provided files.
        file_reader_args : tuple or dict
            Arguments to be supplied to file_reader function.
        data_function : function or None
            For 'function' data_read_type, function must be provided.
        data_function_args : tuple or dict
            Arguments for provided data_function
        multiple_samples : boolean
            If True, there are multiple samples per file, so expand out data list.
        data_read_postprocessing_fn : function or None
            Apply funciton to data once read. Mainly used for combining data
            from different files into a single data structure. E.g., two h5 arrays into one.
        data_keys : string
            Key strings associated with each data stream read, to easily handle
            data and model flow.
        '''
        for data_key in self.data_keys:
            data_read_args = dataset_params[data_key]
            data = self.read_data(**data_read_args)

            self.data[data_key]['data'] = data
            self.data[data_key]['data_batch_transform'] = None
            if type(data_read_args['data_batch_transform']) == str:
                data_batch_transform = self.check_and_get_attr(dataset_functions, data_read_args['data_batch_transform'])
                self.data[data_key]['data_batch_transform'] = data_batch_transform
                self.data[data_key]['data_batch_transform_args'] = data_read_args['data_batch_transform_args']
        
    
    def read_data(self, data_read_type, data_path=None, file_reader=None,
                  file_reader_args={}, data_function=None, data_function_args=None,
                  multiple_samples_per_file=False, data_read_postprocessing_fn=None,
                  **_):
        '''
        Read data from a single input stream.
        '''
        if type(file_reader) == str:
            file_reader = self.check_and_get_attr(dataset_functions, file_reader)
            
        if type(data_function) == str:
            data_function = self.check_and_get_attr(dataset_functions, data_function)
            
        if type(data_read_postprocessing_fn) == str:
            data_read_postprocessing_fn = self.check_and_get_attr(dataset_functions, data_read_postprocessing_fn)
            
        data = self.read_data_handler(data_read_type, data_path, file_reader,
                                      file_reader_args, data_function, data_function_args,
                                      multiple_samples_per_file)
        
        if data_read_postprocessing_fn is not None:
            data = data_read_postprocessing_fn(data)
            
        return data
    
    
    def read_data_handler(self, data_read_type, data_path=None, file_reader=None,
                  file_reader_args={}, data_function=None, data_function_args={},
                  multiple_samples_per_file=False):
        '''
        Data reading handler, for data_read_types:
            ('folder', 'h5', 'file', 'function')
        '''
        assert data_read_type in self.SUPPORTED_DATA_READ_TYPES, (f'{data_read_type}'+
                f' is not supported, only {self.SUPPORTED_DATA_READ_TYPES} are supported.')
        
        if data_read_type == 'folder':
            data = self.read_data_from_folder(data_path, file_reader, file_reader_args,
                                              multiple_samples_per_file)
        
        elif data_read_type == 'h5':
            data = self.read_data_from_file(data_path, h5py.File, ('r',),
                                            multiple_samples_per_file)
            # TODO: Assumes h5's first key is used to access data.
            print('h5 file read, first key is used to access data.')
            data = [d[list(d.keys())[0]][:] for d in data]
            data = data[0]
        elif data_read_type == 'file':
            data = self.read_data_from_file(data_path, file_reader, file_reader_args,
                                            multiple_samples_per_file)
            
        elif data_read_type == 'function':
            data = self.read_data_from_function(data_function, data_function_args)
            
        return data
    
    
    def read_data_from_folder(self, data_path, file_reader, file_reader_args={},
                              multiple_samples_per_file=False):
        '''
        Reads data from all files within folder(s). Ultimately a wrapper for
        read_data_from_file.
        
        data_path : string or list/tuple of strings
            Path(s) to folders to read data from.
        file_reader : function
            Function used to read files in folder.
        file_Reader_args : tuple or dict
            Arguments to be supplied to file_reader function.
        '''
        assert data_path is not None, 'Path to folder(s) must be provided.'
        
        # Read all filenames.
        files = []
        if type(data_path) is str:
            assert os.path.exists(data_path), f'{data_path} does not exist.'
            files.extend([os.path.join(data_path, file) for file in os.listdir(data_path)
                          if os.path.isfile(os.path.join(data_path, file))])
        else:
            assert type(data_path) in (list, tuple), 'data_path must be one of (str, list, tuple).'
            # Multiple data_paths provided, get filenames from all folders.
            for data_p in data_path:
                files.extend([os.path.join(data_p, file) for file in os.listdir(data_p)
                              if os.path.isfile(os.path.join(data_p, file))])
        
        # Now read data with all filenames.
        files = tuple(files)
        data = self.read_data_from_file(files, file_reader, file_reader_args,
                                        multiple_samples_per_file)
        
        return data
    
    
    def read_data_from_file(self, data_path, file_reader, file_reader_args={},
                            multiple_samples_per_file=False):
        '''
        Reads specific file(s).
        
        data_path : string or list/tuple of strings
            Path(s) to file(s) to read data from.
        file_reader : function
            Function used to read and parse files.
        file_reader_args : tuple
            Tuple of arguments.
        multiple_samples_per_file : boolean
            If True, there are multiple samples per file, so expand out data list.
        '''
        if file_reader_args is None:
            file_reader_args = ()
            
        assert data_path is not None, 'Path to folder(s) must be provided.'
        assert file_reader is not None, 'A file reader must be provided.'
        assert type(file_reader_args) in (tuple, dict, list), ('file_reader_args must be either '+
                f'(tuple, dict, list), not {type(file_reader_args)}.')
        
        if type(data_path) is str:
            # Single file to read.
            assert os.path.exists(data_path), f'{data_path} does not exist.'
            if type(file_reader_args) in (list, tuple):
                data = [file_reader(data_path, *tuple(file_reader_args))]
            elif type(file_reader_args) == dict:
                data = [file_reader(data_path, **file_reader_args)]
        
        else:
            assert type(data_path) in (list, tuple), 'data_path must be one of (str, list, tuple).'
            # Multiple data_paths provided, iterate through and do recursive call.
            data = []
            for data_p in data_path:
                file_data = self.read_data_from_file(data_p, file_reader, file_reader_args,
                                                     multiple_samples_per_file)
                data.extend(file_data)
                
        return data
    
    
    def read_data_from_function(self, data_function, data_function_args={}):
        '''
        Reads/generates data using the prescribed data_function and accompanying
        data_function_args.
        
        data_function : function or None
            For 'function' data_read_type, function must be provided.
        data_function_args : tuple or dict
            Arguments for provided data_function.
        '''
        assert type(data_function_args) in (tuple, dict), ('data_function_args must be either '+
                f'(tuple, dict), not {type(data_function_args)}.')
        
        if type(data_function_args) == tuple:
            data = data_function(*data_function_args)
        else:
            data = data_function(**data_function_args)
            
        return data
    
    
    def check_and_get_attr(self, module_name, string):
        '''
        Checks if module has an attribute (function/class) with name equivalent
        to string. If it exists, return that function, otherwise return None.
        '''
        if hasattr(module_name, string):
            attribute = getattr(module_name, string)
        else:
            attribute = None
            
        return attribute
        












