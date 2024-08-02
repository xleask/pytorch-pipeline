# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 21:12:20 2024

Provided functionality for converting cfg architecture into PyTorch layers and
allowing custom layers to be used.

@author: Scott
"""

import inspect
import sys

import torch
from torch import nn

# Used for checking if custom layers in this module have been defined.
this_module = sys.modules[__name__]


class TorchLayersFromList():
    '''
    Class for converting list of dicts, each element specifying a PyTorch layer,
    in order, and its layer parameters. Used to facilitate easy alteration to 
    network design.
    '''
    def __init__(self, input_size, layers_list, layer_name_prefix=None,
                 skip_connections={}, skip_output_cat={}, **_):
        '''
        input_size : tuple
            Input shape/dimensions for the very first ayer. Each subsequent
            shape is inferred.
        layers_list : list of dict
            Submodel layer format in PySubmodel in model.py. Follows form:
                layers_list = {
                    'layer_type': layer_type,
                    'layer_params': layer_params,
                    }
            where layer_type is a string of the exact PyTorch layer to use, and
            layer_params are the keyword arguments of the corresponding layer.
        layer_name_prefix : string or None
            Prefix to add to layer name. In general, this should be given for
            clarity of skip connections.
        skip_connections : dict
            If skip connections are concatenated, need to log size of connected
            output to update future layer sizes.
        skip_output_cat : dict
            For skip connections that are concatenated from other submodels,
            the output size of the layer is affected, so must be updated. Of form:
                {layer_name: (size, to, concatenate)}
        '''
        self.input_size = input_size
        self.layers_list = layers_list
        self.layer_name_prefix = layer_name_prefix
        
        self.skip_connections = skip_connections
        self.skip_output_cat = skip_output_cat
        
        self.layers = nn.ModuleList()
        
        # For tracking/changing input size for each subsequent layer.
        self.layer_input_size = list(input_size)
        
        # Log submodel input size if skip connected to future layer.
        self.handle_skip_connections(f'{layer_name_prefix}_input')
        
        for i, layer_info in enumerate(layers_list):
            torch_layer = self.get_torch_layer(layer_info, i)
            self.layers.append(torch_layer)
            
        self.output_size = self.layer_input_size
        
        
        
    def get_torch_layer(self, layer_info, layer_index):
        '''
        Converts PyNet-read layer (e.g., from cfg) into PyTorch layer.
        
        layer_info : dict
            Follows form:
                layers_list = {
                    'layer_type': layer_type,
                    'layer_params': layer_params,
                    }
            where layer_type is a string of the exact PyTorch layer to use, and
            layer_params are the keyword arguments of the corresponding layer.
        layer_index : integer
            Index of current layer, used for naming.
        '''
        layer_type = layer_info['layer_type']
        layer_params = layer_info['layer_params']
        
        if layer_params is None:
            layer_params = {}
            
        # Check layer_type exists in torch.nn or within this module.
        if hasattr(nn, layer_type):
            layer_fn = getattr(nn, layer_type)
        elif hasattr(this_module, layer_type):
            layer_fn = getattr(this_module, layer_type)
        else:
            assert 0, f'{layer_type} does not exist in torch.nn or layers.py'
            
        # Some layers require single input dimension to be provided, others full
        # input shape, others nothing. This roughly captures all possibilities,
        # but may require tweaking for new/more unique layers.
        layer_fn_kwargs = inspect.signature(layer_fn).parameters.keys()
        if (('in_channels' in layer_fn_kwargs) or ('in_features' in layer_fn_kwargs) or
            ('num_features' in layer_fn_kwargs)):
            input_info = (self.layer_input_size[0],)
        elif ('normalized_shape' in layer_fn_kwargs):
            input_info = (self.layer_input_size,)
        else:
            input_info = ()
            
        torch_layer = layer_fn(*input_info, **layer_params)
        
        # Calculate output_shape/input shape for next layer, by passing in
        # random input. Requires a 'batch' dimension to be added/removed.
        random_input = torch.rand(size=(1, *self.layer_input_size))
        output_size = list(torch_layer(random_input).shape[1:])
        torch_layer.output_size = output_size # Log output size for each layer.
        self.layer_input_size = output_size
        
        # Assign unique layer name to layer.
        if self.layer_name_prefix is not None:
            torch_layer.name = f'{self.layer_name_prefix}_{layer_type}_{layer_index}'
        else:
            torch_layer.name = f'{layer_type}_{layer_index}'
            
        self.handle_skip_connections(torch_layer.name)
        
        return torch_layer
    
    
    def handle_skip_connections(self, layer_name):
        '''
        Handler for layer input/output size changes due to concatenation of
        skip connections.
        '''
        # Increase channel dimension if a previous layer concatenates here.
        if layer_name in self.skip_output_cat:
            # Handle multiple concatenations.
            for skip_output_size in self.skip_output_cat[layer_name]:
                self.layer_input_size = [self.layer_input_size[0]+skip_output_size[0],
                                         *self.layer_input_size[1:]]
        
        # Store layer size as it skip connects with concatenation to a future layer.
        if layer_name in self.skip_connections:
            for future_layer, connection_type in zip(self.skip_connections[layer_name]['future_layers'],
                                                     self.skip_connections[layer_name]['connection_types']):
                if connection_type == 'cat':
                    if future_layer in self.skip_output_cat:
                        self.skip_output_cat[future_layer].append(self.layer_input_size)
                    else:
                        self.skip_output_cat[future_layer] = [self.layer_input_size]
                        

class ResidualBlock(nn.Module):
    '''
    Basic residual connection block.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', device=None,
                 dtype=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out
        
        
        
        
        
        