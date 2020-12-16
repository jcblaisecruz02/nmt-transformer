import torch 
import torch.nn as nn
import numpy as np 
import os
from os import path
from tqdm import tqdm
import warnings

try:
    from apex import amp 
    APEX_AVAILABLE = True 
except ModuleNotFoundError:
    APEX_AVAILABLE = False

def train(model, criterion, optimizer, train_loader, device=None, clip=0.0, scheduler=None, fp16=False):
    train_loss = AverageMeter()
    model.train()

    with tqdm(train_loader) as pbar:
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            out, _ = model(x, y[:, :-1])
            loss = criterion(out.contiguous().flatten(0, 1), 
                            y[:, 1:].contiguous().flatten(0))
                    
            optimizer.zero_grad()

            # Handle backprop, then step opt and scheduler
            if fp16 and APEX_AVAILABLE:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward() 
                    nn.utils.clip_grad_norm_(amp.master_params(optimizer), clip)
            else: 
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            if scheduler is not None: scheduler.step()

            # Set progress bar information
            pbar.set_postfix({'lr': optimizer.param_groups[0]['lr'], 
                              'loss': train_loss.avg})

            train_loss.update(loss.item(), x.shape[0]) 
    return train_loss.avg

def evaluate(model, criterion, valid_loader, device=None):
    valid_loss = AverageMeter()
    model.eval()

    for x, y in tqdm(valid_loader):
        with torch.no_grad():
            x, y = x.to(device), y.to(device)

            out, _ = model(x, y[:, :-1])
            loss = criterion(out.contiguous().flatten(0, 1), 
                            y[:, 1:].contiguous().flatten(0))

            valid_loss.update(loss.item(), x.shape[0])
    return valid_loss.avg

def save_checkpoint(model, args, optimizer=None, e=None, scheduler=None, save_state=False):
    if path.exists(args.save_dir): os.system('rm -r ' + args.save_dir)
    os.mkdir(args.save_dir)

    with open(args.save_dir + '/model.bin', 'wb') as f:
        torch.save(model.state_dict(), f)
    with open(args.save_dir + '/settings.bin', 'wb') as f:
        torch.save([args.hidden_dim, args.n_layers, args.n_heads, args.pf_dim, args.dropout, args.src_msl, args.trg_msl, args.tie_weights], f)
    
    # Save the optimizer state for training
    if save_state and optimizer is not None:
        with warnings.catch_warnings(): # Ignore initial "please save opt and scheduler" warning
            warnings.simplefilter('ignore')
            if scheduler is not None: opt = {'opt_state': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'e': e}
            else: opt = {'opt_state': optimizer.state_dict(), 'scheduler': None, 'e': e}
            with open(args.save_dir + '/training.bin', 'wb') as f:
                torch.save(opt, f)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, fmt=':f'):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
