# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 21:11:57 2024

Load and evaluate trained model; produce plots.

@author: Scott
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from dataset_functions import example_data_reader
from train import import_model_and_model_flow
from utils.cfg_parse import ConfigParser



class Evaluator():
    '''
    Example basic evaluator class for evaluating a trained model.
    '''
    def __init__(self, model, model_params):
        self.model = model
        self.model.eval()
        
        model_dir = os.path.dirname(model_params['architecture_cfg_path'])
        self.results_dir = os.path.sep.join(['results']+model_dir.split(os.path.sep)[1:])
        self.checkpoint = model_params['model_checkpoint_filename']
        self.checkpoint_dir = model_params['model_checkpoint_dir']
        
        
    def __call__(self, evaluation_params, data_path):
        '''
        General handler of evaluation, using evaluation_params read from
        evaluation configuration (yaml) file.
        '''
        self.save_params = evaluation_params['save_params']
        self.save_params['save_dir'] = os.path.join(self.results_dir, self.checkpoint_dir)
        
        inputs = example_data_reader(data_path)
        outputs = self.model(inputs['image_data'], for_inference=True)
        outputs = outputs.detach().numpy()
        outputs *= 255
        outputs = np.clip(outputs, 0, 255)
        outputs = np.uint8(outputs)
        
        self.plot_outputs(outputs)
        
        
    def plot_outputs(self, outputs):
        plt.figure()
        fig, ax = plt.subplots(*(4, 4), figsize=(10, 10), dpi=100)
        
        for i in range(4):
            for j in range(4):
                ax[i][j].imshow(outputs[i*4+j][0])
                
        fig.tight_layout()
        plt.subplots_adjust(hspace=0.25, wspace=0.25)
        
        if self.save_params['save_correction']:
            if not os.path.exists(self.save_params['save_dir']):
                os.makedirs(self.save_params['save_dir'], exist_ok=True)
                
            save_path = os.path.join(self.save_params['save_dir'], self.save_params['save_filename'])
            print('output_file: ', save_path)
            plt.savefig(save_path)
            
            
    
def main(args):
    # Extract training-related parameters.
    training_cfg_file = args['training_cfg']
    training_cfg_parser = ConfigParser(training_cfg_file)
    training_params = training_cfg_parser.params['training']
    model_params = training_cfg_parser.params['model']
    training_save_params = training_cfg_parser.params['saving']
    
    # Extract evaluation-related parameters.
    evaluation_cfg_file = args['evaluation_cfg']
    evaluation_cfg_parser = ConfigParser(evaluation_cfg_file)
    evaluation_params = evaluation_cfg_parser.params['evaluation']
    model_params['model_checkpoint_filename'] = os.path.join(training_save_params['save_folder_name'],
                                                             evaluation_params['model_checkpoint_filename'])
    model_params['model_checkpoint_dir'] = training_save_params['save_folder_name']
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu) for gpu in training_params['gpus']])
    training_params['gpus'] = tuple([i for i in range(len(training_params['gpus']))])
    model_params['gpus'] = training_params['gpus']
    model = import_model_and_model_flow(model_params)
    
    data_path = training_cfg_parser.params['dataset']['gen_inputs']['data_path'][0]
    
    evaluator = Evaluator(model, model_params)
    evaluator(evaluation_params, data_path)
    


if __name__ == '__main__':
    print('\nPyTorch evaluation.\n')
    parser = argparse.ArgumentParser(description=('PyTorch evaluation.\n'))
    
    parser.add_argument('-t', '--training_cfg', type=str, nargs='?',
                        help='String, Path to training config file.')
    parser.add_argument('-e', '--evaluation_cfg', type=str, nargs='?',
                        help='String, Path to evaluation config file.')
    
    args = vars(parser.parse_args())
    
    main(args)
    
    
    