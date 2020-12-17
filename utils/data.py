import torch
import os
import subprocess

def segment(s, spm_model, spm_vocab):
    '''Converts a sentence into segmented wordpieces. Uses split vocab.'''
    cmd = 'echo "{}" | spm_encode --model={} --vocabulary={} --extra_options=bos:eos --output_format=piece'.format(s, spm_model, spm_vocab)
    res = subprocess.check_output(cmd, shell=True, encoding='utf-8').strip()
    return res

def unsegment(s, spm_model):
    '''Converts a sentence into segmented wordpieces'''
    cmd = 'echo "{}" | spm_decode --model={} --input_format=piece'.format(s, spm_model)
    res = subprocess.check_output(cmd, shell=True, encoding='utf-8').strip()
    return res

def pad_and_truncate(s, word2idx, msl, vocab, pad_token='<pad>'):
    '''Takes an unsplit pretokenized sentence and indexes it. Uses joint vocab.'''
    s = [word2idx[token if token in vocab else '<unk>'] for token in s.split()][:msl]
    if len(s) < msl:
        s += [word2idx[pad_token] for _ in range(msl - len(s))]
    return s

def detokenize(tokens, idx2word, word2idx, pad_token='<pad>'):
    '''Takes an indexed sequence and converts them back to segmented wordpieces. Uses joint vocab.'''
    pad_idx = word2idx[pad_token]
    if type(tokens) is torch.Tensor:
        tokens = list(tokens.cpu().numpy())
    tokens = tokens[:tokens.index(pad_idx)]
    return ' '.join([idx2word[idx] for idx in tokens])

def produce_vocabulary(spm_vocab):
    with open(spm_vocab, 'r') as f:
        idx2word = [l.strip().split()[0] for l in f]
        word2idx = {idx2word[i]: i for i in range(len(idx2word))}
        vocab = set(idx2word)
        vocab_sz = len(vocab)
    return idx2word, word2idx, vocab, vocab_sz

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
        self.src_vocab = set(self.src_idx2word)
        self.trg_vocab = set(self.trg_idx2word)
    
    def __getitem__(self, idx):
        with open(self.datadir + '/' + '{}.txt'.format(idx), 'r') as f:
            src, trg = [l.strip() for l in f]
        src = pad_and_truncate(src, self.src_word2idx, self.src_msl, self.src_vocab)
        trg = pad_and_truncate(trg, self.trg_word2idx, self.trg_msl, self.trg_vocab)
        return src, trg
    
    def __len__(self):
        return len(self.data_files)
