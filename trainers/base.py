from itertools import cycle
import json
import numpy as np
import os
import typing as tp

import gin 
import torch
import torch.nn as nn
from torch.optim import Adam 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..metrics import compute_metrics
from .utils import *

@gin.configurable(module='train', denylist=['run_path', 'device'])
class TrainerBase:
    def __init__(
        self, 
        run_path: str, 
        device: str | int, 
        losses, 
        lr, 
        num_steps, 
        log_steps, 
        val_steps, 
        save_steps
    ) -> None:
        self.run_path = run_path 
        self.logger: SummaryWriter = None
        self.device = device
        if isinstance(device, int):
            if device==-1:
                self.device='gpu'
            else:
                self.device = f'cuda:{device}'
        self.eval_metadata_path = os.path.join(run_path, 'eval.json')
        self.losses = losses
        self.losses_logs = {loss_name:[] for loss_name in self.losses.keys()}
        self.lrr = lr
        self.num_steps = num_steps
        self.log_steps = log_steps
        self.val_steps = val_steps
        self.save_steps = save_steps
            
    def init_training(self):
        os.makedirs(self.run_path, exist_ok=True)
        self.logger = SummaryWriter(self.run_path)
    
    @torch.no_grad()
    def val_step(self, step, model: nn.Module, val_loader, metrics):
        model.eval()
        eval_data = compute_metrics(model, self.device, val_loader, 
                                    metrics)
        for metric_name, metric_val in eval_data.items():
            self.logger.add_scalar(f'Eval/{metric_name}', metric_val, global_step=step)

        with open(self.eval_metadata_path, 'r') as f:
            eval_metadata = json.load(f)
        eval_metadata[step] = eval_data
        with open(self.eval_metadata_path, 'w') as f:
            json.dump(eval_metadata, f)
        
    def train_step(self, x, labels, model, optimizer):
        model = model.train()
        x = x.to(self.device)
        labels = labels.to(self.device)
        preds = model(x)
        tot_loss = 0
        for loss_name, loss_fn in self.losses.items():
            l = loss_fn(preds, labels, model)
            self.losses_logs[loss_name].append(l.item())
            tot_loss+=l
        self.losses_logs['Total loss'] = self.losses_logs.get('Total loss', [])+[tot_loss.item()]
        model.zero_grad()
        tot_loss.backward()
        optimizer.step()
    
    @torch.no_grad()
    def log_step(self, step, epoch=None):
        for loss_name, values in self.losses_logs.items():
            self.logger.add_scalar(f'Train/{loss_name}', np.mean(values), global_step=step)
        self.logger.add_scalar('Train/step', step, global_step=step)
        self.logger.add_scalar('Train/epoch', epoch, global_step=step)
        self.losses_logs = {loss_name: [] for loss_name in self.losses_logs.keys()}
    
    @torch.no_grad()     
    def save_state(self, 
                   filename, 
                   model, 
                   optimizer:  tp.Optional[torch.optim.Optimizer] = None, 
                  ):
        if not filename.endswith('.pt'):
            filename+='.pt'
        filename = os.path.join(self.run_path, filename)
               
        if optimizer is not None:
            torch.save({'model': model.state_dict(), 
                        'optim': optimizer.state_dict()}, 
                       filename)
        else:
            torch.save({'model': model.state_dict()}, 
                       filename)

    def log_config(self):
        config = gin.operative_config_str()
        config = config.split('\n')
        config = ['```'] + config + ['```']
        config = '\n'.join(config)
        self.logger.add_text('Config', config, global_step=0)
 
    def fit(self, model, train_loader, val_loader):
        model = model.to(self.device)
        n_epochs = self.num_steps//len(train_loader)
        model.train()
        optimizer = make_optimizer(model)
        is_config_logged = False
        epoch = 0
        
        for step in tqdm(range(self.num_steps), desc=f'Train. epoch {epoch}/{n_epochs}, step {step+1}/{self.num_steps}'):
            epoch = (step//len(train_loader))+1
            x, label = next(cycle(train_loader))
            self.train_step(x, label, model, optimizer)
            
            if not is_config_logged:
                self.log_config()
                is_config_logged = True
            
            if not (step+1)%self.log_steps:
                self.log_step(step+1, epoch)
                
            if not (step+1)%self.val_steps:
                self.val_step(step+1, model, val_loader)

            if not (step+1)%self.save_steps:
                self.save_state(os.path.join(self.run_path, f'{step+1}.pt'), model, optimizer)
        
        self.save_state(os.path.join(self.run_path, f'final.pt'), model)