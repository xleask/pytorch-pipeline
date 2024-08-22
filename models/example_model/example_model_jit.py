# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 21:42:32 2024

Generic (example) PyTorch model.

@author: Scott
"""

import os

import torch

from model_base import PyNetIO

        

class PyNet(PyNetIO):
    '''
    Generic PyTorch network. This class covers high-level behaviours by combining
    PySubmodels which act as simple networks. PyNet allows for grouping PySubmodels
    thereby enabling more complex designs. Cannot have skip connections between
    submodels; create more submodels to accomodate such a design.
    
    The high-level PyNet design must be specified in code. All submodels have the
    same flow (simple feedforward residual design), whose layers and connections
    can be read from a cfg via self.read_architecture_from_cfg.
    '''
    def __init__(self):
        '''
        input_size : tuple or tuple of tuples
            Input dimension of network if a tuple, otherwise input dimensions of
            each input path.
        '''
        super(PyNet, self).__init__()
        
        # Buffer for holding global (cross-submodel) skip connections.
        self.global_skip_outputs = {}
        
        self.float()
    
    
    def forward(self, inputs, for_inference=True, for_discriminator=False):
        '''
        Network call, passing inputs to return output.
        Individual submodel outputs (ref_output/dist_output/etc.) are kept and
        exposed from the model, mainly used in case loss functions depend on
        these intermediate outputs.
        
        inputs : dict or array or list of arrays
            Input(s) to pass to network submodel(s).
        '''
        assert self.submodels_initialized, ('PyNet submodels have not been initialized.' +
            ' Read architecture from cfg using self.read_architecture_from_cfg.')
        
        # Handle inference case for generator and discriminator.
        if for_inference:
            if for_discriminator:
                outputs = self.forward_inference_discriminator(inputs)
            else:
                outputs = self.forward_inference(inputs)
                
            return outputs
        
        # Handle generator.
        gen_outputs = self.forward_generators(inputs['gen_inputs'])
        
        # Handle discriminator.
        disc_outputs = self.forward_discriminators(inputs['gen_inputs'], gen_outputs)
        
        # Reset global skip output buffer.
        self.global_skip_outputs.clear()
        
        outputs = {
            'inputs': inputs,
            'gen_outputs': gen_outputs,
            'disc_outputs': disc_outputs,
            }
        
        return outputs
    
    @torch.no_grad()
    def forward_inference(self, inputs):
        new_skip_dict = {}
        
        generator = self.model_components['generator']
        gen_outputs = generator['gen_cnn'](inputs, new_skip_dict)
        
        # Reset global skip output buffer.
        self.global_skip_outputs.clear()
        
        return gen_outputs
    
    
    def forward_generators(self, inputs):
        new_skip_dict = {}
        
        generator = self.model_components['generator']
        gen_outputs = generator['gen_cnn'](inputs, new_skip_dict)
        
        # Reset global skip output buffer.
        self.global_skip_outputs.clear()
        
        return gen_outputs
    
    @torch.no_grad()
    def forward_inference_discriminator(self, inputs):
        return inputs
    
    
    def forward_discriminators(self, inputs, gen_outputs):
        new_skip_dict = {}
        
        discriminator = self.model_components['discriminator']
        disc_outputs_real = discriminator['disc_cnn'](inputs, new_skip_dict)
        disc_outputs_fake = discriminator['disc_cnn'](gen_outputs, new_skip_dict)
        
        # Reset global skip output buffer.
        self.global_skip_outputs.clear()
        
        disc_outputs = {
            'real': disc_outputs_real,
            'fake': disc_outputs_fake,
            }
        
        return disc_outputs
    


def get_model_instance(architecture_cfg_path, model_checkpoint_filename=None,
                       optimizer=None, gpus=None, **_):
    '''
    Return an instance of the PyNet model whose architecture is read from a cfg.
    
    architecture_cfg_path : string
        Path to PyNet model architecture, specifying, in YAML format, all of the layers and layer
        parameters of each PySubmodel.
    
    For loading:
        
    model_checkpoint_filename : string or None
        If string, filename of model checkpoint to load for model. Only the file
        name is needed, the directory is inferred from architecture_cfg_path.
    optimizer : torch.optim.Optimizer or None
        If given, read state_dict for the optimizer.
    gpus : tuple of ints or None
        GPU device(s) (number(s)) to load model onto, otherwise onto CPU.
    '''
    assert os.path.exists(architecture_cfg_path), f'{architecture_cfg_path} does not exist.'
    model = PyNet()
    model.read_architecture_from_cfg(architecture_cfg_path)
    
    if model_checkpoint_filename is not None:
        optimizer_load = model.load_model(model_checkpoint_filename, optimizer, gpus)
        
        if optimizer is not None:
            return model, optimizer_load
        
    return model
        










