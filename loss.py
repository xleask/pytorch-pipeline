# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 21:12:24 2024

Generic custom loss function for PyNet.

@author: Scott
"""

import sys

import torch
from torch import nn

# Used for checking if custom layers in this module have been defined.
this_module = sys.modules[__name__]


class PyLoss(nn.Module):
    '''
    Loss class for handling different loss functions for training PyNet.
    '''
    def __init__(self, loss_fn, loss_args=()):
        super(PyLoss, self).__init__()
        
        if loss_fn is None:
            self.loss_fn = self.__call__
        elif hasattr(self, loss_fn):
            self.loss_fn = getattr(self, loss_fn)
        elif hasattr(this_module, loss_fn):
            self.loss_fn = getattr(this_module, loss_fn)
        else:
            assert 0, f'{loss_fn} is not defined in loss.py'
            
        # Default loss_args if none provided.
        self.loss_args = {
            'target_adv': 0,
            }
        self.loss_args = self.loss_args | loss_args
        
        
    def cast_to_device(self, device):
        self.to(device)
        
        
    def __call__(self, outputs, train_config=None):
        '''
        outputs :
            Outputs of model.
        train_config : string or None
            If string, one of ('gen', 'disc') for adversarial training.
        '''
        loss = self.loss_fn(outputs, train_config)
        return loss
    
    
    def example_loss(self, outputs, train_config):
        loss = 0
        if self.loss_args['target_adv']:
            target_adv_loss = self.calculate_adversarial_loss(outputs['disc_outputs']['real'],
                                                              outputs['disc_outputs']['fake'],
                                                              train_config=train_config)
            loss += self.loss_args['target_adv'] * target_adv_loss
            
        return loss
    
    
    def calculate_recon_loss(self, pred, target, loss_type='l1'):
        '''
        Basic pixel-wise/element-wise reconstruction loss.
        
        loss_type : string
            One of ('l1', 'l2')
        '''
        if loss_type == 'l1':
            recon_loss_fn = nn.L1Loss()
        else:
            recon_loss_fn = nn.MSELoss()
            
        recon_loss = recon_loss_fn(pred, target)
        return recon_loss
    
    
    def calculate_adversarial_loss(self, disc_real_output, disc_fake_output, train_config, *_):
        '''
        Basic GAN loss function.
        '''
        squeeze_dims = tuple([i+1 for i in range(len(disc_real_output.shape[1:]))])
        disc_real_output = torch.squeeze(disc_real_output, squeeze_dims)
        disc_fake_output = torch.squeeze(disc_fake_output, squeeze_dims)
        
        criterion = nn.BCELoss()
        
        real_label = 1.0
        fake_label = 0.0
        
        if train_config == 'disc':
            label1 = torch.full((disc_real_output.shape[0],), real_label, dtype=torch.float,
                                device=disc_real_output.device)
            errD_real = criterion(disc_real_output, label1)
            label2 = torch.full((disc_fake_output.shape[0],), fake_label, dtype=torch.float,
                                device=disc_real_output.device)
            errD_fake = criterion(disc_fake_output, label2)
            return errD_real + errD_fake
        else:
            label = torch.full((disc_fake_output.shape[0],), real_label, dtype=torch.float,
                                device=disc_real_output.device)
            errG = criterion(disc_fake_output, label)
            return errG
        
        
def get_loss(loss_params):
    '''
    Return an instance of PyLoss.
    
    loss_args : dict
        {argument:value} arguments of PyLoss class.
    '''
    py_loss = PyLoss(**loss_params)
    return py_loss
















