# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 21:12:28 2024

Base PyNet model functionality.

@author: Scott
"""

import json
import os
import time
import yaml

from torch import nn
import torch

from layers import TorchLayersFromList


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
        

class PyNetIO(nn.Module):
    '''
    Base class that PyNet inherits from, giving all loading and saving
    functionality.
    '''
    def __init__(self):
        super(PyNetIO, self).__init__()
        
        self.loss_history = []
        self.n_epochs_trained = 1
        self.architecture_cfg = None
        self.submodels_initialized = False
        
        
    def read_architecture_from_cfg(self, cfg_path):
        '''
        PyNet submodel architectures must be read from cfg file. This facilitates
        programmatic testing of different designs and editing outside of the codebase.
        
        cfg_path : string
            Path to cfg with PyNet submodel architectures.
        '''
        assert type(cfg_path) == str, f'cfg_path must be string, not {type(cfg_path)}.'
        
        self.architecture_cfg = cfg_path # Keep for saving/loading this architecture.
        _, cfg_format = os.path.splitext(cfg_path)
        
        self.submodels = nn.ModuleDict()
        self.model_components = nn.ModuleDict()
        
        assert cfg_format == '.yaml', 'Only yaml configs are supported.'
        with open(cfg_path) as yaml_file:
            architecture = yaml.safe_load(yaml_file)
            
        self.adv_types = {}
        
        for model_component_key, model_component_data in architecture['model_components'].items():
            adv_type = model_component_data['adv_type']
            submodel_list = model_component_data['submodels']
            zero_out_skips = False
            if 'zero_out_skips' in model_component_data.keys():
                zero_out_skips = model_component_data['zero_out_skips']
                
            submodels = self.read_submodels(architecture, submodel_list, zero_out_skips)
            
            self.adv_types[model_component_key] = adv_type
            component = {model_component_key: submodels}
            self.model_components.update(component)
            
        # Buffer was temporarily used for modifying layer sizes due to concatenated skip connections.
        self.global_skip_outputs.clear()
        
        self.submodels_initialized = True
        
        
    def read_submodels(self, architecture, submodel_list, zero_out_skips):
        '''
        Reads submodel information from cfg and creates PySubmodel.
        '''
        submodels = nn.ModuleDict()
        for submodel_name in submodel_list:
            # Configure submodel input.
            submodel_input = architecture[submodel_name]['submodel_input']
            submodel_input_keys = list(submodel_input.keys())
            
            if 'input_size' in submodel_input_keys:
                # Submodel input size is explicitly specified.
                input_size = tuple(submodel_input['input_size'])
            elif 'input_data' in submodel_input_keys:
                # Submodel input size specified via data input size.
                input_size = tuple(submodel_input['input_data']['input_size'])
            else:
                # Submodel input size based on output of another submodel.
                submodel_output_key = submodel_input_keys[0]
                if submodel_output_key in submodels:
                    # Submodel is within this model component.
                    input_size = submodels[submodel_input_keys[0]].output_size
                else:
                    # Submodel belongs to another model component, so need to find it.
                    for model_component_key, model_component_data in self.model_components.items():
                        if submodel_output_key in model_component_data.keys():
                            input_size = self.model_components[model_component_key][submodel_output_key].output_size
                            
            submodel_args = architecture[submodel_name]
            submodel = PySubmodel(input_size, self.global_skip_outputs, **submodel_args)
            submodel.zero_out_skips = zero_out_skips
            submodel.apply(weights_init)
            submodels.update({submodel_name: submodel})
            
        return submodels
    
    
    def save_model(self, save_filename_prefix=None, optimizer=None,
                   early_termination=False, save_folder_name=None):
        '''
        Saves model state_dict, optimizer state_dict, and accompanying details.
        
        save_filename_prefix : string or None
            If string, prefix to add at beginning of filename.
        optimizer : torch.optim.Optimizer or None
            If not None, optimizer to save state for.
        early_termination : boolean
            If True, training stopped in the middle of an epoch, so signal this
            with suffix '_et' in filename before date and time.
        save_folder_name : string or None
            If string, save checkpoints to subfolder within model_checkpoints.
        '''
        assert self.architecture_cfg is not None, ('No architecture has been read, ' +
                                                   'so cannot save.')
        architecture_dir = os.path.dirname(self.architecture_cfg)
        checkpoint_dir = os.path.join(architecture_dir, 'model_checkpoints')
        
        if save_folder_name is not None:
            checkpoint_dir = os.path.join(checkpoint_dir, save_folder_name)
            
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        save_filename = self.generate_save_filename(save_filename_prefix, early_termination)
        save_path = os.path.join(checkpoint_dir, save_filename)
        
        if os.path.exists(save_path):
            overwrite_current = 'y' == input('File exists. Overwrite? [y/n]\n')
            if not overwrite_current:
                print('File already exists, not overwriting.')
                return
            
        self._torch_save(save_path, optimizer)
        
        
    def generate_save_filename(self, save_filename_prefix=None, early_termination=False):
        '''
        Generate save filename in form of:
            <save_filename_prefix>(_)epoch<n_epochs_trained>(_et)_<YYYYMMDD-HHMMSS.pth
            
        save_filename_prefix : string or None
            If string, prefix to add at beginning of filename.
        early_termination : boolean
            If True, training stopped in the middle of an epoch, so signal this
            with suffix '_et' in filename before date and time.
        '''
        save_filename = ''
        if save_filename_prefix is not None:
            save_filename += f'{save_filename_prefix}_'
        
        save_filename += f'epoch{self.n_epochs_trained}'
        
        if early_termination:
            save_filename += '_et'
        
        time_str = time.strftime("%Y%m%d-%H%M%S")
        
        # Common PyTorch file extension (or .pth)
        save_filename += f'_{time_str}.pth'
        
        return save_filename
    
    
    def _torch_save(self, save_filename, optimizer=None):
        '''
        Save model and optimizer information to file.
        '''
        if optimizer is None:
            opt_state_dict = None
        else:
            opt_state_dict = optimizer.state_dict()
        
        with open(save_filename, 'wb') as f:
            torch.save({
                'architecture': self.architecture_cfg,
                'epoch': self.n_epochs_trained,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': opt_state_dict,
                'loss_history': self.loss_history,
                }, f)
            
        print(f'Saved model at epoch {self.n_epochs_trained}.')
        
        
    def load_model(self, load_filename, optimizer=None, gpus=None):
        '''
        Load model state_dict, optimizer state_dict, and accompanying details.
        Requires model and optimizer to be initialized in advance.
        
        Returns optimizer if it is given.
        
        load_filename : string
            Filename of model checkpoints in architecture's model_checkpoints folder.
            The correct directory is inferred from reading architecture_cfg.
        optimizer : torch.optim.Optimizer or None
            If given, read state_dict for the optimizer.
        gpus : tuple of ints or None
            GPU device(s) (number(s)) to load model onto, otherwise onto CPU.
        '''
        assert self.architecture_cfg is not None, ('No architecutre has been read, ' +
                                                   'so cannot load.')
        architecture_dir = os.path.dirname(self.architecture_cfg)
        checkpoint_dir = os.path.join(architecture_dir, 'model_checkpoints')
        load_path = os.path.join(checkpoint_dir, load_filename)
        assert os.path.exists(load_path), f'{load_path} does not exist.'
        
        if torch.cuda.is_available() and gpus != None and len(gpus) != 0:
            assert type(gpus) == tuple, f'gpus must be a tuple, not {type(gpus)}.'
            gpu_str = 'cuda'
            device = torch.device(gpu_str)
        else:
            device = torch.device('cpu')
            
        checkpoint = torch.load(load_path, map_location=device)
        
        updated_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            updated_key = key
            # Handle DataParallel training, which prepends 'module.' to state_dict.
            if key[:6] == 'module':
                updated_key = updated_key[7:]
                
            # Handle legacy weights.
            if not key.startswith('model_component'):
                if 'disc' in key:
                    component_type = 'discriminator'
                else:
                    component_type = 'generator'
                    
                updated_key = '.'.join(['model_components', component_type, key])
                
            updated_state_dict[updated_key] = value
            
        checkpoint['model_state_dict'] = updated_state_dict
        
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.loss_history = checkpoint['loss_history']
        self.n_epochs_trained = checkpoint['epoch']
        
        if optimizer is not None and checkpoint['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return optimizer
        
        
    def get_n_params(self):
        '''
        Returns total number of parameters in the full PyNet.
        '''
        total_params = 0
        return 0
        # TODO: Fix this function.
        for submodel in self.submodels:
            total_params += submodel.get_n_params()
            
        return total_params
    
    
    def increment_epochs_trained(self):
        '''
        Used for incrementing PyNetIO.n_epochs_trained.
        '''
        self.n_epochs_trained += 1
        


class PySubmodel(nn.Module):
    '''
    Generic submodel for the PyNet, which is defined through submodel_layers
    (see below) which get converted to corresponding PyTorch layers through
    TorchLayersFromList.
    '''
    def __init__(self, input_size, global_skip_outputs, layers, layer_name_prefix=None,
                 skip_connections={}, **_):
        '''
        input_size : tuple
            Size/shape of input to submodel.
        layers : dict
            Dictionary used to create PyTorch layers from cfg. Of form:
                layers = {
                    'layer_type': layer_type,
                    'layer_params': layer_params,
                    }
            where layer_type is the PyTorch layer and layer_params are the associated
            layer parameters.
        layer_name_prefix : string or None
            The submodel name which is the prefix for all submodel layer names.
        skip_connections : dict
            Dictionary where each key is the layer name whose output is skip connected
            to the output of the layers whose names are given in the corresponding value.
            Also supports taking submodel input to be skip-connected, whose name is
            given by {layer_name_prefix}_input.
            E.g.:
                skip_connections = {
                    'dist_input': ['dist_ReLU_2', 'dist_ReLU_5'],
                    }
        '''
        super(PySubmodel, self).__init__()
        
        self.input_size = input_size
        self.layer_name_prefix = layer_name_prefix
        self.skip_connections = skip_connections
        self.zero_out_skips = False
        
        torch_layer_converter = TorchLayersFromList(input_size, layers,
                                                    layer_name_prefix, skip_connections,
                                                    global_skip_outputs)
        self.layers = torch_layer_converter.layers
        self.output_size = torch_layer_converter.output_size
        
        # Buffers for holding local (intra-submodel) and global (cross-submodel) skip connections.
        self.skip_outputs = {}
        self.global_skip_outputs = global_skip_outputs
        
        self.float()
        
        
    def forward(self, inputs, new_skip_dict=None):
        '''
        Submodel forward call. Does not necessarily need to match self.input_size,
        as some layers are size-agnostic.
        
        inputs : array
            Input to submodel.
        '''
        if new_skip_dict is None:
            new_skip_dict = {}
            
        output = inputs
        
        # Make residual connection from input layer for future layers.
        self.make_residual_connection(f'{self.layer_name_prefix}_input', output, new_skip_dict)
        
        for layer in self.layers:
            output = layer(output)
            output = self.add_residual_connection(layer.name, output, new_skip_dict)
            
            self.make_residual_connection(layer.name, output, new_skip_dict)
            
        # Reset skip connection buffer.
        self.skip_outputs = {}
        
        return output
    
    
    def make_residual_connection(self, layer_name, output, new_skip_dict):
        '''
        If a residual connection should be made, hold the connection in
        skip_outputs for the future layer whose output it connects to.
        
        layer_name : string
            Name of layer whose output is to be skip connected.
        output : array
            Output of layer_name.
        '''
        if layer_name in self.skip_connections:
            # layer_name's output skip connects to given layers.
            for future_layer, connection_type in zip(self.skip_connections[layer_name]['future_layers'],
                                                     self.skip_connections[layer_name]['connection_types']):
                # If prefix matches, update local skip_output, otherwise global.
                if self.layer_name_prefix in future_layer:
                    skip_dict = self.skip_outputs
                else:
                    skip_dict = self.global_skip_outputs
                    
                skip_dict = new_skip_dict
                # Checks if future_layer has been signalled before.
                if future_layer in skip_dict:
                    skip_dict[future_layer].append((output, connection_type))
                else:
                    skip_dict[future_layer] = [(output, connection_type)]
                    
        
    def add_residual_connection(self, layer_name, output, new_skip_dict):
        '''
        When at a future layer of residual connection, add all skip outputs to
        this layer's output.
        
        layer_name : string
            Name of layer to check/add skipped connections to its output.
        output : array
            Output of layer_name.
        '''
        # Add residual connection.
        if layer_name in new_skip_dict:
            # layer_name has residual connection(s) to add, so add them all.
            for (skip_output, connection_type) in new_skip_dict[layer_name]:
                if connection_type == 'add':
                    output = output + skip_output
                else:
                    output = torch.cat([output, skip_output], dim=1)
                    
            # Free up memory
            del self.skip_outputs[layer_name]
            
        #Add global residual connection.
        if layer_name in self.global_skip_outputs:
            # layer_name has residual connection(s) to add, so add them all.
            for (skip_output, connection_type) in self.global_skip_outputs[layer_name]:
                if connection_type == 'add':
                    output = output + skip_output
                else:
                    output = torch.cat([output, skip_output], dim=1)
                    
        return output
    
    
    def get_n_params(self):
        '''
        Returns total number of parameters in the given submodel.
        '''
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return n_params
            
        
















