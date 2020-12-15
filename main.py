import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os
from os import path
from tqdm import tqdm
import argparse

from utils.data import TextDataset, collate_fn
from utils.model import Encoder, Decoder, Seq2Seq, count_parameters

def train(model, criterion, optimizer, train_loader, device=None, clip=0.0, scheduler=None):
    train_loss = 0
    model.train()
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)

        out, _ = model(x, y[:, :-1])
        loss = criterion(out.contiguous().flatten(0, 1), 
                        y[:, 1:].contiguous().flatten(0))
                
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler is not None: scheduler.step()

        train_loss += loss.item()
    train_loss /= len(train_loader) 
    return train_loss

def evaluate(model, criterion, valid_loader, device=None):
    valid_loss = 0
    model.eval()
    for x, y in tqdm(valid_loader):
        with torch.no_grad():
            x, y = x.to(device), y.to(device)

            out, _ = model(x, y[:, :-1])
            loss = criterion(out.contiguous().flatten(0, 1), 
                            y[:, 1:].contiguous().flatten(0))

            valid_loss += loss.item()
    valid_loss /= len(valid_loader) 
    return valid_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--valid_dir', type=str)
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--src_vocab', type=str)
    parser.add_argument('--trg_vocab', type=str)
    parser.add_argument('--src_msl', type=int, default=100)
    parser.add_argument('--trg_msl', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=1)

    parser.add_argument('--save_dir', type=str)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--pf_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--clip', type=float, default=0.1)
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')

    parser.add_argument('--seed', type=int, default=1111)
    
    args = parser.parse_args()
    print(args)

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

        # Produce training setup
        encoder = Encoder(vocab_sz=train_dataset.src_vocab_sz, hidden_dim=args.hidden_dim, n_layers=args.n_layers, n_heads=args.n_heads, pf_dim=args.pf_dim, dropout=args.dropout, msl=args.src_msl)
        decoder = Decoder(vocab_sz=train_dataset.trg_vocab_sz, hidden_dim=args.hidden_dim, n_layers=args.n_layers, n_heads=args.n_heads, pf_dim=args.pf_dim, dropout=args.dropout, msl=args.trg_msl)
        model = Seq2Seq(encoder, decoder, train_dataset.src_word2idx['<pad>'], train_dataset.trg_word2idx['<pad>']).to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.src_word2idx['<pad>'])
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * args.epochs)

        print("Model has {:,} trainable parameters.".format(count_parameters(model)))

        # Train Model
        print("Beginning Training.")
        for e in range(1, args.epochs + 1):
            train_loss = train(model, criterion, optimizer, train_loader, device=device, clip=args.clip, scheduler=scheduler)
            valid_loss = evaluate(model, criterion, valid_loader, device=device)
            print("Epoch {:3} | Train Loss {:.4f} | Train Ppl {:.4f} | Valid Loss {:.4f} | Valid Ppl {:.4f}".format(e, train_loss, np.exp(train_loss), valid_loss, np.exp(valid_loss)))

        # Save the model and states
        print("Saving checkpoint.")
        if path.exists(args.save_dir):
            print("Save directory exists. Deleting.")
            os.system('rm -r ' + args.save_dir)
        model = model.cpu()
        os.mkdir(args.save_dir)
        with open(args.save_dir + '/model.bin', 'wb') as f:
            torch.save(model.state_dict(), f)
        with open(args.save_dir + '/settings.bin', 'wb') as f:
            torch.save([args.hidden_dim, args.n_layers, args.n_heads, args.pf_dim, args.dropout, args.src_msl, args.trg_msl], f)

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
            hd, nl, nh, pf, dp, smsl, tmsl = torch.load(f)
        print(test_dataset.src_vocab_sz)
        encoder = Encoder(vocab_sz=test_dataset.src_vocab_sz, hidden_dim=hd, n_layers=nl, n_heads=nh, pf_dim=pf, dropout=dp, msl=smsl)
        decoder = Decoder(vocab_sz=test_dataset.trg_vocab_sz, hidden_dim=hd, n_layers=nl, n_heads=nh, pf_dim=pf, dropout=dp, msl=tmsl)
        model = Seq2Seq(encoder, decoder, test_dataset.src_word2idx['<pad>'], test_dataset.trg_word2idx['<pad>'])

        with open(args.save_dir + '/model.bin', 'rb') as f:
            model.load_state_dict(torch.load(f))
        model = model.to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=test_dataset.src_word2idx['<pad>'])
        
        print("Begin testing.")
        test_loss = evaluate(model, criterion, test_loader, device=device)
        print("Test Loss {:.4f} | Test Ppl {:.4f}".format(test_loss, np.exp(test_loss)))
        
        

if __name__ == '__main__':
    main()