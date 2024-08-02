# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 21:12:15 2024

Performs JIT tracing of a given model, required for profiling.
The input model must be JIT friendly, i.e., avoids constant argument inputs
or using dictionaries as returned outputs.

@author: Scott
"""

import argparse
import os

import torch

from train import import_model_and_model_flow, ConfigParser


def jit_trace_model(model, input_size, jit_output_path, save_input=False):
    device = 'cpu'
    dummy_input = torch.randn(*input_size).to(device)
    model_jit_trace = torch.jit.trace(model, dummy_input)
    
    os.makedirs(os.path.dirname(jit_output_path), exist_ok=True)
    
    torch.jit.save(model_jit_trace, jit_output_path)
    print(f'Saved JIT traced model to {jit_output_path}')
    
    if save_input:
        raw_input_path = os.path.splitext(jit_output_path)[0] + '.raw'
        dummy_input.numpy().tofile(raw_input_path)
        print(f'Saved JIT dummy input to {raw_input_path}')
        
        
def create_jit_output_path(model_params, checkpoint_filename):
    architecture_dir = os.path.dirname(model_params['architecture_cfg_path'])
    checkpoint_dir = os.path.join(architecture_dir, 'model_checkpoints')
    jit_dir = os.path.join(checkpoint_dir, os.path.dirname(model_params['model_checkpoint_filename']), 'jit')
    
    # Replace fullt stops in filename.
    jit_prefix = os.path.splitext(checkpoint_filename)[0].replace('.', '_')
    jit_filename = jit_prefix + '_jit.pt'
    jit_output_path = os.path.join(jit_dir, jit_filename)
    return jit_output_path


def main(args):
    # Read training params
    training_cfg_file = args['training_cfg']
    training_cfg_parser = ConfigParser(training_cfg_file)
    training_params = training_cfg_parser.params['training']
    model_params = training_cfg_parser.params['model']
    training_save_params = training_cfg_parser.params['saving']
    
    checkpoint_path = args['checkpoint']
    checkpoint_filename = os.path.basename(checkpoint_path)
    model_params['model_checkpoint_filename'] = os.path.join(training_save_params['save_folder_name'],
                                                             checkpoint_path)
    model_params['model_checkpoint_dir'] = training_save_params['save_folder_name']
    # Assume JIT-friendly model is defined by a file with the same model name
    # as for training but with a '_jit' suffix.
    model_flow_path_filename, model_flow_path_ext = os.path.splitext(model_params['model_flow_path'])
    model_params['model_flow_path'] = f'{model_flow_path_filename}_jit{model_flow_path_ext}'
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu) for gpu in training_params['gpus']])
    training_params['gpus'] = tuple([i for i in range(len(training_params['gpus']))])
    model_params['gpus'] = training_params['gpus']
    model = import_model_and_model_flow(model_params)
    
    input_size = args['input_size']
    save_input = args['save_input']
    
    jit_output_path = create_jit_output_path(model_params, checkpoint_filename)
    jit_trace_model(model, input_size, jit_output_path, save_input)


if __name__ == '__main__':
    print('\nPyTorch model JIT tracing.\n')
    parser = argparse.ArgumentParser(description=('PyTorch model JIT tracing.\n'))
    
    parser.add_argument('-t', '--training_cfg', type=str, nargs='?',
                        help='String, Path to training config file.')
    parser.add_argument('-c', '--checkpoint', type=str, nargs='?',
                        help='String, Path to model checkpoint for jit tracing. Only need filename.')
    parser.add_argument('-i', '--input_size', type=int, nargs='*',
                        help='Tuple of Integers, Input size (including batch size) of model.')
    
    # Optional args
    parser.add_argument('-si', '--save_input', type=bool, nargs='?', default=True, const=True,
                        help='Boolean, Also save dummy input in raw format.')
    
    args = vars(parser.parse_args())
    
    main(args)
    




