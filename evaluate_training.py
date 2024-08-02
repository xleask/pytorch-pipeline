# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 21:12:02 2024

Evaluate training convergence behaviours of various models.

@author: Scott
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.cfg_parse import ConfigParser


class TrainingEvaluator():
    '''
    Class for evaluating training of models, comparing loss convergence.
    '''
    def __init__(self):
        pass
    
    
    def __call__(self, evaluation_params):
        '''
        General handler of training evaluation, using parameters read from
        training evaluation configuration (yaml) file.
        '''
        self.checkpoint_dir = evaluation_params['checkpoint_dir']
        checkpoints_to_evaluate = evaluation_params['checkpoints']
        saving_params = evaluation_params['saving_params']
        preprocessing_params = evaluation_params['preprocessing_params']
        
        checkpoint_data = self.read_all_checkpoints(self.checkpoint_dir, checkpoints_to_evaluate)
        checkpoint_data = self.preprocess_data(checkpoint_data, preprocessing_params)
        self.plot_loss_histories(checkpoint_data, saving_params)
    
    
    def read_all_checkpoints(self, checkpoint_dir, checkpoints):
        checkpoint_data = []
        for checkpoint_filename in checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
            data = self.read_checkpoint_data(checkpoint_path)
            checkpoint_data.append(data)
            
        return checkpoint_data
    
    
    def read_checkpoint_data(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        
        checkpoint_loss_history = checkpoint['loss_history']
        checkpoint_optimizer = checkpoint['optimizer_state_dict']
        
        data = {
            'loss': checkpoint_loss_history,
            'optimizer': checkpoint_optimizer,
            }
        return data
    
    
    def preprocess_data(self, checkpoint_data, preprocessing_params):
        for data in checkpoint_data:
            data['loss'] = self.preprocess_loss(data['loss'], preprocessing_params['loss_average_window'])
            data['label'] = self.create_checkpoint_label(data)
            
        return checkpoint_data
    
    
    def preprocess_loss(self, loss_history, loss_average_window):
        '''
        Averages over loss_average_window values, non-overlapping.
        '''
        new_loss_history = []
        n_loss = int(np.ceil(len(loss_history)/loss_average_window))
        for i in range(n_loss):
            new_loss = np.mean(loss_history[i*100:i*100+100])
            new_loss_history.append(new_loss)
            
        return new_loss_history
    
    
    def create_checkpoint_label(self, data):
        initial_lr = data['optimizer']['param_groups'][0]['initial_lr']
        final_lr = data['optimizer']['param_groups'][0]['lr']
        label = f'{initial_lr}_{final_lr}'
        return label
    
    
    def plot_loss_histories(self, checkpoint_data, saving_params):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=200)
        
        markers = (None, 'o', '^', 's', 'd')
        marker_every = 5
        mark_every = 100
        for i, data in enumerate(checkpoint_data):
            marker = markers[i//marker_every]
            ax.plot(data['loss'], marker=marker, markevery=mark_every, label=data['label'])
            
        ax.legend(loc='upper right', frameon=False)
        ax.set_ylabel('L1 Loss')
        ax.set_xlabel('Iterations/100')
        ax.set_yscale('log')
        fig.tight_layout()
        plt.subplots_adjust(hspace=0.25, wspace=0.25)
        
        if saving_params['save_output']:
            save_dir = self.checkpoint_dir
            save_dir = save_dir.replace('models', 'results')
            save_dir = save_dir.replace('model_checkpoints', '')
            filename = os.path.join(save_dir, saving_params['filename'])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            print('output_file: ', filename)
            plt.savefig(filename)
            
     
def main(args):
    evaluation_loss_cfg_file = args['training_evaluation_cfg']
    evaluation_loss_cfg_parser = ConfigParser(evaluation_loss_cfg_file)
    
    evaluation_params = evaluation_loss_cfg_parser.params['evaluation']
    
    evaluator = TrainingEvaluator()
    evaluator(evaluation_params)
       
     

if __name__ == '__main__':
    print('\nPyTorch training evaluation.\n')
    parser = argparse.ArgumentParser(description=('PyTorch training evaluation.\n'))
    
    parser.add_argument('-te', '--training_evaluation_cfg', type=str, nargs='?',
                        help='String, Path to training evaluation config file.')
    
    args = vars(parser.parse_args())
    
    main(args)
        








