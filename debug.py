# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 21:11:49 2024

Module for running debugging on model training or evaluation.

@author: Scott
"""

from train import main
from evaluate import main as main_evaluation

#%% Training debug

args = {'training_cfg':r'configs/example_model/example_training_cfg.yaml'}
main(args)

#%% Evaluation debug

args = {
        'training_cfg':r'config/example_model/example_training_cfg.yaml',
        'evaluation_cfg':r'config/example_model/example_evaluation_cfg.yaml'
        }
main_evaluation(args)