# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 21:12:37 2024

Base trainer class for handling most basic training set up and operations.

@author: Scott
"""

import os

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch



class ModelTrainerBase():
    '''
    Class for handling training model, training loop, and monitoring training
    statistics.
    '''
    def __init__(self, model, loss_fn, py_dataset, optimizer, batch_size=32,
                 learning_rate=0.01, learning_rate_gamma=0.2, learning_rate_step=3,
                 gpus=None, adversarial_args=None, **_):
        '''
        model : PyNet instance
            Initialized/loaded PyNet model to train with.
        loss_fn : PyLoss instance
            Loss function to minimize.
        py_dataset : PyDataset instance
            Dataset for loading and feeding data to model for training.
        optimizer : torch.optim.Optimizer
            Optimizer to use for training.
        batch_size : int
            Number of samples to use per batch.
        learning_rate : float
            Optimizer learning rate.
        learning_rate_gamma : float
            Learning rate gamma/decay rate.
        learning_rate_step_size : integer
            Number of epochs before applying learning_rate_gamma.
        gpus : tuple of ints or None
            GPU device(s) (number(s)) to load model onto, otherwise onto CPU.
        '''
        self.gpus = gpus
        if torch.cuda.is_available() and self.gpus != None and len(self.gpus) != 0:
            assert type(self.gpus) == tuple, f'gpus must be a tuple, not {type(gpus)}.'
            if len(self.gpus) > 1:
                # Handle distributed training
                self.gpu_id = int(os.environ['LOCAL_RANK'])
                model = model.to(self.gpu_id)
                model = DDP(model, device_ids=[self.gpu_id], find_unused_parameters=True)
                device = self.gpu_id
            else:
                self.gpu_id = 0
                gpu_str = 'cuda'
                device = torch.device(gpu_str)
                model = model.to(device)
                model = torch.nn.DataParallel(model, device_ids=self.gpus)
        else:
            self.gpu_id = 0
            device = torch.device('cpu')
            model = model.to(device)
            
        model.train()
        
        dataloader = self.prepare_dataloader(py_dataset, batch_size=batch_size, shuffle=True)
        
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=learning_rate_step,
                                                       gamma=learning_rate_gamma)
        self.learning_rate = learning_rate
        self.learning_rate_gamma = learning_rate_gamma
        self.learning_rate_step = learning_rate_step
        self.model = model
        loss_fn = loss_fn.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.dataset = py_dataset
        self.lr_scheduler = lr_scheduler
        
        self.prepare_adversarial_training(adversarial_args)
        
        
    def prepare_adversarial_training(self, adversarial_args):
        '''
        Prepare adversarial training if enabled.
        '''
        if adversarial_args is not None and adversarial_args['enable_adversarial_training']:
            self.adversarial_training = True
            disc_lr_scheduler = torch.optim.lr_scheduler.StepLR(adversarial_args['optimizer'],
                                                                step_size=adversarial_args['learning_rate_step'],
                                                                gamma=adversarial_args['learning_rate_gamma'])
            self.disc_lr_scheduler = disc_lr_scheduler
            self.disc_optimizer = adversarial_args['optimizer']
            self.train_disc_first = adversarial_args['train_disc_first']
            
            # Switch training between generator and discriminator. Track specific
            # training progress.
            self.n_gen_iterations = adversarial_args['n_gen_iterations']
            self.n_disc_iterations = adversarial_args['n_disc_iterations']
            self.current_config = 'disc' if self.train_disc_first else 'gen'
            self.config_iter_counter = 0
            
            self.gen_loss = 'N/A'
            self.disc_loss = 'N/A'
            self.gen_loss_format = '>3'
            self.disc_loss_format = '>3'
            
            self.adversarial_args = adversarial_args
        else:
            self.adversarial_training = False
            
        
    def get_adversarial_iter_config(self):
        '''
        If adversarial training is enabled, selects the correct optimizer
        and LR scheduler, and handles weight freezing/training.
        
        Otherwise return only optimizer and scheduler.
        '''
        if hasattr(self.model, 'module'):
            model = self.model.module
        else:
            model = self.model
        
        if not self.adversarial_training:
            return self.optimizer, self.lr_scheduler
        
        config_change = False
        if self.current_config == 'gen' and self.config_iter_counter >= self.n_gen_iterations:
            self.current_config = 'disc'
            self.config_iter_counter = 0
            config_change = True
        elif self.current_config == 'disc' and self.config_iter_counter >= self.n_disc_iterations:
            self.current_config = 'gen'
            self.config_iter_counter = 0
            config_change = True
            
        if self.current_config == 'gen':
            optimizer = self.optimizer
            lr_scheduler = self.lr_scheduler
        else:
            optimizer = self.disc_optimizer
            lr_scheduler = self.disc_lr_scheduler
            
        if config_change or self.config_iter_counter == 0:
            self.prepare_adversarial_weights(model)
            
        self.config_iter_counter += 1
        
        return optimizer, lr_scheduler
    
    
    def prepare_adversarial_weights(self, model):
        '''
        For adversarial training, require swapping between training/freezing
        generator and discriminator. This method does this at the beginning of
        each epoch.
        '''
        if self.current_config == 'gen':
            gen_requires_grad = True
            disc_requires_grad = False
        else:
            gen_requires_grad = False
            disc_requires_grad = True
            
        for component_key, component_submodels in model.model_components.items():
            if model.adv_types[component_key] == 'generator':
                requires_grad = gen_requires_grad
            else:
                requires_grad = disc_requires_grad
                
            for param in component_submodels.parameters():
                param.requires_grad = requires_grad
                
    
    def prepare_dataloader(self, py_dataset, batch_size, shuffle=True):
        '''
        Function to prepare dataloader for feeding data from PyDataset instance
        to model.
        
        py_dataset : PyDataset
            Instance of PyDataset
        batch_size : integer
            Number of samples in the batch.
        '''
        if len(self.gpus) > 1:
            dataloader = DataLoader(
                py_dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=DistributedSampler(py_dataset),
                )
        else:
            dataloader = DataLoader(py_dataset, batch_size=batch_size, shuffle=True)
            
        return dataloader
    
    
    def print_losses(self, batch_number, n_batches, loss, print_every_n=1):
        '''
        Handle printing losses to console.
        
        batch_number : integer
            Current iteration/batch number in epoch.
        print_every_n : integer
            Only print every print_every_n iterations.
        '''
        if self.gpu_id != 0:
            return
        
        sci_notation_thr = 0.0001
        current_batch = batch_number + 1
        
        if batch_number % print_every_n == 0:
            batch_str = f'[{current_batch:>3d}/{n_batches:>3d}]'.ljust(self.epoch_header_width)
            
            if loss < sci_notation_thr:
                loss_format = '.2e'
            else:
                loss_format = '>6f'
                
            if self.adversarial_training:
                if self.current_config == 'gen':
                    self.gen_loss = loss
                    self.gen_loss_format = loss_format
                else:
                    self.disc_loss = loss
                    self.disc_loss_format = loss_format
                    
                gen_loss_str = f'{self.gen_loss:{self.gen_loss_format}}'.ljust(self.epoch_header_width)
                disc_loss_str = f'{self.disc_loss:{self.disc_loss_format}}'.ljust(self.epoch_header_width)
                print(self.console_left_pad, f'{gen_loss_str}{disc_loss_str}{batch_str}', end='\r', flush=True)
            else:
                loss_str = f'{loss:{loss_format}}'.ljust(self.epoch_header_width)
                print(self.console_left_pad, f'{loss_str}{batch_str}', end='\r', flush=True)
                
    
    def print_loss_header(self, epoch):
        '''
        Prints loss header to console.
        
        epoch : integer
            Current epoch number.
        '''
        if self.gpu_id != 0:
            return
        
        self.console_left_pad = '\t'
        self.epoch_header_width = 16
        epoch_str = f'Epoch {epoch}'
        underline_str = '-'*len(epoch_str)
        print('\n')
        print(self.console_left_pad, epoch_str.ljust(self.epoch_header_width, ' '))
        print(self.console_left_pad, underline_str.ljust(self.epoch_header_width, ' '))
        
        if self.adversarial_training:
            gen_loss_str = 'Gen Loss'.ljust(self.epoch_header_width)
            disc_loss_str = 'Disc Loss'.ljust(self.epoch_header_width)
            print(self.console_left_pad, f'{gen_loss_str}{disc_loss_str}Batch Number')
        else:
            loss_str = "Loss".ljust(self.epoch_header_width)
            print(self.console_left_pad, f'{loss_str}Batch Number')
            
    
def get_optimizer(params, optimizer, optimizer_args=()):
    '''
    Get PyTorch optimizer.
    
    params : PyTorch parameters
        Weights to be optimized.
    optimizer : None or str or torch.optim.Optimizer
        If None, creates default Adam optimizer.
        If str, creates optimizer with that PyTorch name and corresponding
        optimizer_args.
        If torch.optim.Optimizer, just returns the instance.
    '''
    if optimizer is None:
        # Optimizer is not given, so use default.
        optimizer = torch.optim.Adam(params, betas=(0.5, 0.999), weight_decay=0.01, *optimizer_args)
    elif type(optimizer) == str:
        # Create optimizer from PyTorch optimizer name.
        if hasattr(torch.optim, optimizer):
            optimizer_fn = getattr(torch.optim, optimizer)
            optimizer = optimizer_fn(params, betas=(0.5, 0.999), weight_decay=0.01, *optimizer_args)
        else:
            assert hasattr(torch.optim, optimizer), f'PyTorch has no optimizer {optimizer}.'
            
    else:
        assert isinstance(optimizer, torch.optim.Optimizer), 'Provided optimizer is invalid.'
        
    return optimizer













