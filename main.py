import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel

import numpy as np
import argparse
import math

from utils.data import TextDataset, collate_fn
from utils.model import Encoder, Decoder, Seq2Seq, count_parameters
from utils.train import train, evaluate, save_checkpoint
from utils.optim import NoamLR, LabelSmoothingLoss

try:
    from apex import amp 
    APEX_AVAILABLE = True 
except ModuleNotFoundError:
    APEX_AVAILABLE = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, help='Directory of training samples')
    parser.add_argument('--valid_dir', type=str, help='Directory of validation samples')
    parser.add_argument('--test_dir', type=str, help='Directory of test samples')
    parser.add_argument('--src_vocab', type=str, help='SentencePiece vocabulary file for source sentence')
    parser.add_argument('--trg_vocab', type=str, help='SentencePiece vocabulary file for target sentence')
    parser.add_argument('--src_msl', type=int, default=100, help='Maximum sequence length for source sentence')
    parser.add_argument('--trg_msl', type=int, default=100, help='Maximum sequence length for target sentence')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel workers for dataloading')
    parser.add_argument('--save_dir', type=str, help='Directory to save checkpoints and load checkpoints from')

    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--tie_weights', action='store_true', help='Tie weights of encoder/decoder embeddings and projection layer')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimensions of the transformer layers')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of transformer blocks in the encoder and decoder')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--pf_dim', type=int, default=512, help='Positionwise feedforward projection dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping')

    parser.add_argument('--criterion', type=str, default='cross_entropy', choices=['cross_entropy', 'label_smoothing'], help='Criterion to use')
    parser.add_argument('--smoothing', type=float, default=0.0, help='Label smoothing factor')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'lamb'], help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for non LayerNorm and Bias layers')
    parser.add_argument('--adam_epsilon', type=float, default=1e-9, help='Epsilon value for Adam')
    parser.add_argument('--adam_b1', type=float, default=0.9, help='Beta1 for LAMB')
    parser.add_argument('--adam_b2', type=float, default=0.99, help='Beta2 for LAMB')
    parser.add_argument('--scheduler', type=str, default=None, choices=['cosine', 'linear', 'noam', None], help='Scheduler to use')
    parser.add_argument('--warmup_pct', type=float, default=0.1, help='Percentage of steps to warmup for linear scheduler')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='Number of warmup steps for noam scheduling')
    
    parser.add_argument('--do_train', action='store_true', help='Train a model')
    parser.add_argument('--do_test', action='store_true', help='Evaluate a model')
    parser.add_argument('--resume_training', action='store_true', help='Toggle to resume from checkpoint in --save_dir')
    parser.add_argument('--no_cuda', action='store_true', help='Do not use GPU')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 training via APEX')
    parser.add_argument('--opt_level', type=str, default='O1', choices=['O1', 'O2'], help='Optimization level for FP16 training')
    parser.add_argument('--save_every', type=int, default=1, help='Save checkpoint every --save_every epoch')
    parser.add_argument('--pad_token', type=str, default='<pad>', help='Override default padding token')

    parser.add_argument('--use_swa', action='store_true', help='Use stochastic weight averaging')
    parser.add_argument('--swa_pct', type=float, help='Last percentage of total training steps to average')
    parser.add_argument('--swa_times', type=int, help='Number of times to average over swa_pct')

    parser.add_argument('--seed', type=int, default=1111, help='Random seed')
    
    args = parser.parse_args()
    print(args)

    # Set seeds
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    if args.do_train:
        # Produce dataloaders
        print("Producing dataloaders.")
        train_dataset = TextDataset(args.train_dir, args.src_vocab, args.trg_vocab, src_msl=args.src_msl, trg_msl=args.trg_msl)
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                  sampler=train_sampler,
                                                  batch_size=args.batch_size, 
                                                  collate_fn=collate_fn, 
                                                  num_workers=args.num_workers)

        valid_dataset = TextDataset(args.valid_dir, args.src_vocab, args.trg_vocab, src_msl=args.src_msl, trg_msl=args.trg_msl)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                                                  shuffle=False,
                                                  batch_size=args.batch_size, 
                                                  collate_fn=collate_fn, 
                                                  num_workers=args.num_workers)

        print("Training batches: {}\nValidation batches: {}".format(len(train_loader), len(valid_loader)))

        # Produce model and criterion
        encoder = Encoder(vocab_sz=train_dataset.src_vocab_sz, 
                          hidden_dim=args.hidden_dim, 
                          n_layers=args.n_layers, 
                          n_heads=args.n_heads, 
                          pf_dim=args.pf_dim, 
                          dropout=args.dropout, 
                          msl=args.src_msl, 
                          fp16=args.fp16)
        decoder = Decoder(vocab_sz=train_dataset.trg_vocab_sz, 
                          hidden_dim=args.hidden_dim, 
                          n_layers=args.n_layers, 
                          n_heads=args.n_heads, 
                          pf_dim=args.pf_dim, 
                          dropout=args.dropout, 
                          msl=args.trg_msl, 
                          fp16=args.fp16)
        model = Seq2Seq(encoder, decoder, train_dataset.src_word2idx[args.pad_token], train_dataset.trg_word2idx[args.pad_token], tie_weights=args.tie_weights).to(device)
        
        # Configure SWA
        if args.use_swa: swa_model = AveragedModel(model)
        else: swa_model = None

        # Produce criterion
        if args.criterion == 'cross_entropy':
            criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.src_word2idx[args.pad_token])
        elif args.criterion == 'label_smoothing':
            criterion = LabelSmoothingLoss(epsilon=args.smoothing)

        # Produce Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
                                         "weight_decay": args.weight_decay}, 
                                        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
                                         "weight_decay": 0.0}]

        if args.optimizer == 'adamw':
            try:
                from transformers import AdamW
                optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(args.adam_b1, args.adam_b2))
            except ModuleNotFoundError:
                print("Transformers module not found for AdamW. Using generic Adam instead.")
                optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(args.adam_b1, args.adam_b2))
        elif args.optimizer == 'lamb':
            try:
                from pytorch_lamb import Lamb
                optimizer = Lamb(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(args.adam_b1, args.adam_b2))
            except ModuleNotFoundError:
                print("LAMB implementation not found. Using generic Adam instead.")
                optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(args.adam_b1, args.adam_b2))
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(args.adam_b1, args.adam_b2))

        # Configure FP16
        if args.fp16 and APEX_AVAILABLE:
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

        # Produce the scheduler
        if args.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * args.epochs)
        elif args.scheduler == 'linear':
            try:
                from transformers import get_linear_schedule_with_warmup
                steps = args.epochs * len(train_loader)
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(steps * args.warmup_pct), num_training_steps=steps)
            except ModuleNotFoundError:
                print('Transformers module not found for Linear Schedule. Not using a scheduler instead.')
                scheduler = None
        elif args.scheduler == 'noam':
            scheduler = NoamLR(optimizer, warmup_steps=args.warmup_steps)
        else:
            scheduler = None

        print("\nUsing {} optimizer with {} scheduling. Optimizing via {}.".format(str(type(optimizer)), str(type(scheduler)), str(type(criterion))))
        print("Model has {:,} trainable parameters.\n".format(count_parameters(model)))

        # Configure states if resuming from checkpoint
        if args.resume_training:
            print("Loading from checkpoint...", end='')
            with open(args.save_dir + '/model.bin', 'rb') as f:
                model.load_state_dict(torch.load(f))
                print('Model loaded...', end='')
            if args.use_swa:
                with open(args.save_dir + '/swa_model.bin', 'rb') as f:
                    swa_model.load_state_dict(torch.load(f))
                    print('SWA Model loaded...', end='')
            with open(args.save_dir + '/training.bin', 'rb') as f:
                training_state = torch.load(f)
                optimizer.load_state_dict(training_state['opt_state'])
                e = training_state['e'] + 1 # Start on the next epoch
                print('Optimizer loaded...', end='')

                if training_state['scheduler'] is not None:
                    scheduler.load_state_dict(training_state['scheduler'])
                    print('Scheduler loaded...', end='')
                else:
                    print('No scheduler found...')
            global_steps = len(train_loader) * (e - 1)
            print("Done!\nLoaded checkpoint from epoch {} | Global step {}!".format(training_state['e'], global_steps))
            
        # Else, begin from epoch 1
        else:
            print("Beginning training from epoch 1.")
            e = 1
            global_steps = 0

        # Print training setup
        total_steps = len(train_loader) * (args.epochs)
        print('Total number of steps: {}'.format(total_steps))
        
        # Configure SWA points
        if args.use_swa: 
            swa_every = sorted(list(set([round(total_steps * (1 - args.swa_pct * i)) for i in range(args.swa_times)])))
            print('SWA on steps: {}\n'.format(swa_every)) 
        else: 
            swa_every = None
            print('\n')

        # Train Model
        while e < args.epochs + 1:
            # Train one epoch
            train_loss, global_steps = train(model, criterion, optimizer, train_loader, global_steps, 
                                            device=device, clip=args.clip, scheduler=scheduler, fp16=args.fp16, 
                                            swa=args.use_swa, swa_every=swa_every, swa_model=swa_model)
            valid_loss = evaluate(model, criterion, valid_loader, device=device)

            print("Epoch {:3} | Train Loss {:.4f} | Train Ppl {:.4f} | Valid Loss {:.4f} | Valid Ppl {:.4f}".format(e, train_loss, np.exp(train_loss), valid_loss, np.exp(valid_loss)))

            # Save the checkpoint
            if e % args.save_every == 0 or e == args.epochs:
                print('Saving checkpoint for epoch {}...'.format(e), end='')
                save_checkpoint(model, args, optimizer=optimizer, e=e, scheduler=scheduler, save_state=True, swa_model=swa_model)
                print('Done!')
            
            # Update epoch number
            e += 1

        # Evaluate again and save if we're using SWA
        if args.use_swa:
            print('Evaluating on final averaged model.')
            valid_loss = evaluate(swa_model, criterion, valid_loader, device=device)
            print("Valid Loss {:.4f} | Valid Ppl {:.4f}".format(valid_loss, np.exp(valid_loss)))
            print('Saving checkpoint for averaged model.')
            save_checkpoint(model, args, optimizer=optimizer, e=e, scheduler=scheduler, save_state=True, swa_model=swa_model)
            print('Done!')

        print("Training done!\n")

    if args.do_test:    
        # Produce dataloaders
        print("Producing test loaders.")
        test_dataset = TextDataset(args.test_dir, args.src_vocab, args.trg_vocab, src_msl=args.src_msl, trg_msl=args.trg_msl)
        test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                  shuffle=False,
                                                  batch_size=args.batch_size, 
                                                  collate_fn=collate_fn, 
                                                  num_workers=args.num_workers)
        
        print("Number of testing batches: {}".format(len(test_loader)))

        # Produce setup
        print("Loading model and saved settings.")
        with open(args.save_dir + '/settings.bin', 'rb') as f:
            hd, nl, nh, pf, dp, smsl, tmsl, tw, usw, cri = torch.load(f)

        encoder = Encoder(vocab_sz=test_dataset.src_vocab_sz, 
                          hidden_dim=hd, 
                          n_layers=nl, 
                          n_heads=nh, 
                          pf_dim=pf, 
                          dropout=dp, 
                          msl=smsl, 
                          fp16=args.fp16)
        decoder = Decoder(vocab_sz=test_dataset.trg_vocab_sz, 
                          hidden_dim=hd, 
                          n_layers=nl, 
                          n_heads=nh, 
                          pf_dim=pf, 
                          dropout=dp, 
                          msl=tmsl, 
                          fp16=args.fp16)
        model = Seq2Seq(encoder, decoder, test_dataset.src_word2idx[args.pad_token], test_dataset.trg_word2idx[args.pad_token], tie_weights=tw)

        if cri == 'cross_entropy':
            criterion = nn.CrossEntropyLoss(ignore_index=test_dataset.trg_word2idx[args.pad_token])
        elif cri == 'label_smoothing':
            criterion = LabelSmoothingLoss(epsilon=args.smoothing)

        # Load the checkpoint
        if usw:
            swa_model = AveragedModel(model)
            with open(args.save_dir + '/swa_model.bin', 'rb') as f:
                swa_model.load_state_dict(torch.load(f))
                swa_model = swa_model.to(device)

        with open(args.save_dir + '/model.bin', 'rb') as f:
            model.load_state_dict(torch.load(f))
        model = model.to(device)
        
        print("\nBegin testing.")
        test_loss = evaluate(model, criterion, test_loader, device=device)
        print("Test Loss {:.4f} | Test Ppl {:.4f}".format(test_loss, np.exp(test_loss)))

        if usw:
            print('Testing SWA model.')
            test_loss = evaluate(swa_model, criterion, test_loader, device=device)
            print("Test Loss {:.4f} | Test Ppl {:.4f}".format(test_loss, np.exp(test_loss)))

if __name__ == '__main__':
    main()
