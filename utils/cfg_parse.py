# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 21:14:36 2024

Class for parsing training configuration file.

@author: Scott
"""

import yaml


class ConfigParser():
    '''
    Basic YAML configuration parser.
    '''
    def __init__(self, yaml_path):
        '''
        yaml_path : string
            Path to YAML configuration file.
        '''
        self.process_yaml_file(yaml_path)
        
    
    def process_yaml_file(self, yaml_path):
        '''
        Handles simple processing of yaml config file. Considerably more straight
        forward than cfg/configparser route.
        '''
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)
            
        self.params = config_data