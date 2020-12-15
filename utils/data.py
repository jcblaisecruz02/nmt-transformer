import torch
import os

def pad_and_truncate(s, word2idx, msl, pad_token='<pad>'):
    '''Takes an unsplit pretokenized sentence and indexes it'''
    s = [word2idx[token] for token in s.split()][:msl]
    if len(s) < msl:
        s += [word2idx[pad_token] for _ in range(msl - len(s))]
    return s

def detokenize(tokens, idx2word, word2idx, pad_token='<pad>'):
    pad_idx = word2idx[pad_token]
    if type(tokens) is torch.Tensor:
        tokens = list(tokens.cpu().numpy())
    tokens = tokens[:tokens.index(pad_idx)]
    return ' '.join([idx2word[idx] for idx in tokens])

def collate_fn(batch):
    srcs, trgs = [], []
    for b in batch:
        srcs.append(b[0])
        trgs.append(b[1])
    return torch.LongTensor(srcs), torch.LongTensor(trgs)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, datadir, src_vocab, trg_vocab, src_msl, trg_msl):
        self.datadir = datadir
        self.src_msl = src_msl
        self.trg_msl = trg_msl
        self.data_files = os.listdir(datadir)
        
        with open(src_vocab, 'r') as f:
            self.src_idx2word = [l.strip().split()[0] for l in f]
            self.src_word2idx = {self.src_idx2word[i]: i for i in range(len(self.src_idx2word))}
        with open(trg_vocab, 'r') as f:
            self.trg_idx2word = [l.strip().split()[0] for l in f]
            self.trg_word2idx = {self.trg_idx2word[i]: i for i in range(len(self.trg_idx2word))}
            
        self.src_vocab_sz = len(self.src_idx2word)
        self.trg_vocab_sz = len(self.trg_idx2word)
    
    def __getitem__(self, idx):
        with open(self.datadir + '/' + '{}.txt'.format(idx), 'r') as f:
            src, trg = [l.strip() for l in f]
        src = pad_and_truncate(src, self.src_word2idx, self.src_msl)
        trg = pad_and_truncate(trg, self.trg_word2idx, self.trg_msl)
        return src, trg
    
    def __len__(self):
        return len(self.data_files)
