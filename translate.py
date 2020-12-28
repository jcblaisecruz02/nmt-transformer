import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.data import segment, pad_and_truncate, unsegment, produce_vocabulary
from utils.model import Encoder, Decoder, Seq2Seq
from utils.translation import translate_one_sentence

from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--translate_sentence', action='store_true')
    parser.add_argument('--translate_file', action='store_true')
    parser.add_argument('--src_file', type=str)
    parser.add_argument('--output_file', type=str)

    parser.add_argument('--joint_model', type=str)
    parser.add_argument('--joint_vocab', type=str)
    parser.add_argument('--src_vocab', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--sentence', type=str)

    parser.add_argument('--beams', type=int, default=1)
    parser.add_argument('--msl', type=int, default=100)
    parser.add_argument('--desegment', action='store_true')
    parser.add_argument('--max_words', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--is_segmented', action='store_true')
    parser.add_argument('--strategy', type=str, default='bfs')
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.92)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--use_topk', action='store_true')
    parser.add_argument('--pad_token', type=str, default='<pad>')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--use_swa', action='store_true')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Build vocabulary
    idx2word, word2idx, vocab_set, vocab_sz = produce_vocabulary(args.joint_vocab)

    # Load model
    print('Loading saved checkpoint.')
    with open(args.save_dir + '/settings.bin', 'rb') as f:
        hd, nl, nh, pf, dp, smsl, tmsl, tw, usw, cri = torch.load(f)

    encoder = Encoder(vocab_sz, hd, nl, nh, pf, dp, smsl, fp16=False)
    decoder = Decoder(vocab_sz, hd, nl, nh, pf, dp, tmsl, fp16=False)
    model = Seq2Seq(encoder, decoder, word2idx[args.pad_token], word2idx[args.pad_token], tie_weights=tw).to(device)

    if args.use_swa: 
        print("Using averaged model.")
        model = AveragedModel(model)
        with open(args.save_dir + '/swa_model.bin', 'rb') as f:
            model.load_state_dict(torch.load(f))
    else:
        print("Using saved model.")
        with open(args.save_dir + '/model.bin', 'rb') as f:
            model.load_state_dict(torch.load(f))
    model = model.eval()

    # Translate
    if args.translate_sentence:
        print('Beginning translation.')
        out,  attn = translate_one_sentence(args.sentence, model, args.joint_model, args.src_vocab, idx2word, word2idx, vocab_set, 
                                            beams=args.beams, msl=args.msl, desegment=args.desegment, max_words=args.max_words, seed=args.seed, 
                                            device=device, is_segmented=args.is_segmented, strategy=args.strategy, 
                                            top_k=args.top_k, top_p=args.top_p, temperature=args.temperature, use_topk=args.use_topk)
        print(out)

    if args.translate_file:
        print('Loading source file')
        with open(args.src_file, 'r') as f:
            src_sentences = [l.strip() for l in f]
        
        print('Producing translations')
        translations = []
        for s in tqdm(src_sentences):
            out,  attn = translate_one_sentence(s, model, args.joint_model, args.src_vocab, idx2word, word2idx, vocab_set, 
                                                beams=args.beams, msl=args.msl, desegment=args.desegment, max_words=args.max_words, seed=args.seed, 
                                                device=device, is_segmented=args.is_segmented, strategy=args.strategy, 
                                                top_k=args.top_k, top_p=args.top_p, temperature=args.temperature, use_topk=args.use_topk)
            if not args.desegment: out = ' '.join(out)
            #print(out)
            translations.append(out)
        
        print('Writing to file')
        with open(args.output_file, 'w') as f:
            for line in translations:
                f.write(line + '\n')
        print('Done!')

if __name__ == '__main__':
    main()
