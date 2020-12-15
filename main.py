import torch
import torch.nn as nn

import os
from tqdm import tqdm
import argparse

from utils.data import TextDataset, collate_fn
from utils.model import Encoder, Decoder, Seq2Seq

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
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    
    args = parser.parse_args()
    print(args)
    
    if args.do_train:
        pass
    
    if args.do_test:    
        test_dataset = TextDataset(args.test_dir, args.src_vocab, args.trg_vocab, src_msl=args.src_msl, trg_msl=args.trg_msl)
        test_sampler = torch.utils.data.RandomSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                  sampler=test_sampler,
                                                  batch_size=args.batch_size, 
                                                  collate_fn=collate_fn, 
                                                  num_workers=args.num_workers)
        
        print(len(test_loader), len(test_dataset))
        print(test_dataset.src_vocab_sz)
        
        src, trg = next(iter(test_loader))
        print(src.shape, trg.shape)
        print(src)
        
        print('Testing model')
        
        encoder = Encoder(vocab_sz=test_dataset.src_vocab_sz, hidden_dim=10, n_layers=2, n_heads=2, pf_dim=12, dropout=0.1)
        decoder = Decoder(vocab_sz=test_dataset.trg_vocab_sz, hidden_dim=10, n_layers=2, n_heads=2, pf_dim=12, dropout=0.1)
        model = Seq2Seq(encoder, decoder, test_dataset.src_word2idx['<pad>'], test_dataset.trg_word2idx['<pad>'])

        criterion = nn.CrossEntropyLoss(ignore_index=3)
        
        #print(model)
        print()
        
        with torch.no_grad():
            out, _ = model(src, trg[:, :-1])
            loss = criterion(out.contiguous().flatten(0, 1), trg[:, 1:].contiguous().flatten(0))

        print(src.shape, src.contiguous().flatten(0, 1).shape)
        print(trg.shape, trg[:, 1:].contiguous().flatten(0).shape)
        print(out.shape)
            
            
        
        print(loss.item())
        
        

if __name__ == '__main__':
    main()