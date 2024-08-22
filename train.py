# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 21:12:33 2024

Module for training PyNet model.

@author: Scott
"""

import argparse
import importlib
import os

from dataset import get_dataset
from loss import get_loss
from train_base import ModelTrainerBase, get_optimizer
from utils.cfg_parse import ConfigParser

from torch.distributed import init_process_group, destroy_process_group
import torch



class ModelTrainer(ModelTrainerBase):
    '''
    Class for handling training model, training loop, and monitoring training
    statistics.
    '''
    def __init__(self, model, loss_fn, dataset, adversarial_args, training_params, **_):
        super(ModelTrainer, self).__init__(model, loss_fn, dataset, adversarial_args=adversarial_args,
                                           **training_params)
        
    
    def train_model(self, n_epochs, saving_params):
        '''
        Loop through all epochs, calling main training loop and saving the model.
        
        n_epochs : int
            Number of epochs to train for.
        save_filename_prefix : string or None
            If string, save model checkpoints using prefix to add at beginning
            of filename.
        save_epoch_increment : int
            Save model checkpoint every save_epoch_increment epochs.
        '''
        if hasattr(self.model, 'module'):
            model = self.model.module
        else:
            model = self.model
        
        save_filename_prefix = saving_params['save_filename_prefix']
        save_epoch_increment = saving_params['save_epoch_increment']
        save_folder_name = saving_params['save_folder_name']
        
        # Save initial weights.
        if (save_filename_prefix is not None) and (self.gpu_id == 0):
            # Need self.gpu_id == 0 to only save master model in distributed training.
            save_filename_init = save_filename_prefix + '_init_weights'
            model.save_model(save_filename_init, self.optimizer, False,
                             save_folder_name)
            
        for epoch in range(model.n_epochs_trained, model.n_epochs_trained+n_epochs):
            self.print_loss_header(epoch)
            
            if len(self.gpus) > 1:
                # Distributed training, need this to shuffle properly.
                self.dataloader.sampler.set_epoch(epoch)
                
            epoch_loss_history, early_termination = self.training_loop()
            model.loss_history.extend(epoch_loss_history)
            
            if save_filename_prefix is not None:
                if (early_termination or (epoch%save_epoch_increment==0)) and (self.gpu_id == 0):
                    # Need self.gpu_id == to only save master model in distributed training.
                    model.save_model(save_filename_prefix, self.optimizer, early_termination,
                                     save_folder_name)
                    
            if early_termination:
                # User-signalled termination.
                break
            
            model.increment_epochs_trained()
            
    
    def training_loop(self):
        '''
        For agiven epoch, iterate through batched dataset, updating weights and
        collecting iteration losses.
        '''
        # For user-termination, flag early_termination
        early_termination = False
        
        n_batches = len(self.dataloader)
        epoch_loss_history = []
        try:
            for batch_number, input_batch in enumerate(self.dataloader):
                optimizer, lr_scheduler = self.get_adversarial_iter_config()
                optimizer.zero_grad()
                
                outputs = self.model(input_batch)
                
                loss = self.loss_fn(outputs, self.current_config)
                
                total_loss = self.backward_losses(loss)
                optimizer.step()
                
                epoch_loss_history.append(total_loss)
                self.print_losses(batch_number, n_batches, total_loss)
                
        except KeyboardInterrupt:
            print('\n\nKeyboardInterrupt: Stopping training.')
            early_termination = True
            
        if self.adversarial_training:
            self.lr_scheduler.step()
            self.disc_lr_scheduler.step()
        else:
            lr_scheduler.step()
            
        return epoch_loss_history, early_termination
    
    
    def backward_losses(self, loss):
        '''
        Handles retaining of graphs and backward passes for multiple losses,
        if applicable.
        
        loss : PyLoss output or tuple of PyLoss output
            Loss(es) from PyNet.
        '''
        if type(loss) == tuple:
            total_loss = 0
            for j, individual_loss in enumerate(loss):
                if j < len(loss)-1:
                    individual_loss.backward(retain_graph=True)
                else:
                    individual_loss.backward()
                    
                total_loss = total_loss + individual_loss.item()
        else:
            loss.backward(retain_graph=False)
            total_loss = loss.item()
            
        return total_loss
    

def import_model_and_model_flow(model_params):
    '''
    Model architectures (whether same or different) may have different model
    flows that are most easily set in their own model flow Python module.
    Import the specific module here.
    '''
    model_flow_path = model_params['model_flow_path']
    model_flow_file, _ = os.path.splitext(model_flow_path)
    
    # Assume neither Windows/Linux overlap use of slashes. Only one works.
    model_flow_module_str = model_flow_file.replace('\\', '.')
    model_flow_module_str = model_flow_module_str.replace('/', '.')
    
    model_flow = importlib.import_module(model_flow_module_str)
    importlib.reload(model_flow)
    model = model_flow.get_model_instance(**model_params)
    
    return model


def main(args):
    '''
    Main function for setting up training.
    Requires training config file to be provided for parsing training setup.
    '''
    # Parse training parameters
    cfg_file = args['training_cfg']
    cfg_parser = ConfigParser(cfg_file)
    
    saving_params = cfg_parser.params['saving']
    training_params = cfg_parser.params['training']
    adversarial_params = cfg_parser.params['adversarial']
    model_params = cfg_parser.params['model']
    
    # Set up training on specific GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu) for gpu in training_params['gpus']])
    training_params['gpus'] = tuple([i for i in range(len(training_params['gpus']))])
    gpus = training_params['gpus']
    model_params['gpus'] = gpus
    
    # Set up model and optimizer(s)
    model = import_model_and_model_flow(model_params)
    
    gen_params = []
    disc_params = []
    for component_key, component_submodels in model.model_components.items():
        component_params = [p for p in component_submodels.parameters() if p.requires_grad]
        if model.adv_types[component_key] == 'generator':
            gen_params += component_params
        else:
            disc_params += component_params
        
    optimizer = get_optimizer(gen_params, training_params['optimizer'],
                              (training_params['learning_rate'],))
    training_params['optimizer'] = optimizer
    
    if adversarial_params['enable_adversarial_training']:
        disc_optimizer = get_optimizer(disc_params, adversarial_params['optimizer'],
                                       (adversarial_params['learning_rate'],))
        adversarial_params['optimizer'] = disc_optimizer
    
    # Set up loss
    loss_params = cfg_parser.params['loss']
    loss_fn = get_loss(loss_params)
    
    # Set up dataset
    dataset_params = cfg_parser.params['dataset']
    dataset = get_dataset(dataset_params)
    
    # Train
    print('Running PyTorch training...')
    
    # Need to clear the cache to stop memory leaking into GPU 0
    # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
    if len(gpus) > 1:
        # Handle distributed training.
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        torch.cuda.empty_cache()
        init_process_group(backend='nccl')
    else:
        for gpu in gpus:
            torch.cuda.set_device(gpu)
            torch.cuda.empty_cache()
            
    model_trainer = ModelTrainer(model, loss_fn, dataset, adversarial_params,
                                 training_params)
    model_trainer.train_model(n_epochs=training_params['n_epochs'], saving_params=saving_params)
    
    if len(gpus) > 1:
        destroy_process_group()
        
    print('\n\n')
    


if __name__ == '__main__':
    print('\nPyTorch training.\n')
    parser = argparse.ArgumentParser(description=('PyTorch training.\n'))
    
    parser.add_argument('-t', '--training_cfg', type=str, nargs='?',
                        help='String, Path to training config file.')
    
    args = vars(parser.parse_args())
    
    main(args)






