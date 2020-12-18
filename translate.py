import argparse

from utils.model import Encoder, Decoder, Seq2Seq
from utils.data import segment, unsegment, pad_and_truncate, detokenize, produce_vocabulary
from utils.translation import translate, translate_one_sentence

import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--translate_sentence', action='store_true', help='Translate one unsegmented sentence')
    parser.add_argument('--translate_file', action='store_true', help='Translate a file with segmented sequences')
    parser.add_argument('--sentence', type=str, help='Sentence to translate')
    parser.add_argument('--src_file', type=str, help='File to translate')
    parser.add_argument('--output_file', type=str, help='File to write translations to')
    parser.add_argument('--use_swa', action='store_true', help='Use the saved averaged model')

    parser.add_argument('--src_vocab', type=str, help='Isolated vocabulary for the source language')
    parser.add_argument('--joint_vocab', type=str, help='Joint vocabulary for both languages')
    parser.add_argument('--spm_model', type=str, help='Joint sentencepiece model')
    parser.add_argument('--save_dir', type=str, help='Location of saved checkpoint')
    parser.add_argument('--pad_token', type=str, help='Override padding token', default='<pad>')
    parser.add_argument('--msl', type=int, default=100, help='Maximum sequence length of the model')
    parser.add_argument('--max_words', type=int, default=20, help='Maximum number of tokens to generate from the model')
    parser.add_argument('--seed', type=int, default=1111, help='Random seed')
    parser.add_argument('--desegment', action='store_true', help='Desegment the translation')
    parser.add_argument('--use_cuda', action='store_true')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    torch.manual_seed(args.seed)

    # Load vocabularies
    print('Loading vocabularies.')
    idx2word, word2idx, vocab_set, vocab_sz = produce_vocabulary(args.joint_vocab)
    src_idx2word, src_word2idx, src_vocab_set, src_vocab_sz = produce_vocabulary(args.src_vocab)

    # Load model and saved settings
    print('Loading model.')
    with open(args.save_dir + '/settings.bin', 'rb') as f:
        hd, nl, nh, pf, dp, smsl, tmsl, tw, usw, cri = torch.load(f)

    encoder = Encoder(vocab_sz=vocab_sz, 
                      hidden_dim=hd, 
                      n_layers=nl, 
                      n_heads=nh, 
                      pf_dim=pf, 
                      dropout=dp, 
                      msl=smsl, 
                      fp16=False)
    decoder = Decoder(vocab_sz=vocab_sz, 
                      hidden_dim=hd, 
                      n_layers=nl, 
                      n_heads=nh, 
                      pf_dim=pf, 
                      dropout=dp, 
                      msl=tmsl, 
                      fp16=False)
    model = Seq2Seq(encoder, decoder, word2idx[args.pad_token], word2idx[args.pad_token], tie_weights=tw)
    
    # Load checkpoint
    if args.use_swa: 
        print("Using averaged model.")
        model = AveragedModel(model)
        with open(args.save_dir + '/swa_model.bin', 'rb') as f:
            model.load_state_dict(torch.load(f))
    else:
        print("Using saved model.")
        with open(args.save_dir + '/model.bin', 'rb') as f:
            model.load_state_dict(torch.load(f))
    model = model.to(device)
    model.eval()

    if args.translate_sentence:
        # Translate
        print('Producing translation.')
        predictions, _ = translate_one_sentence(args.sentence, 
                                                        model, 
                                                        args.spm_model, 
                                                        args.src_vocab, 
                                                        idx2word, 
                                                        word2idx, 
                                                        vocab_set, 
                                                        msl=args.msl, 
                                                        desegment=args.desegment, 
                                                        max_words=args.max_words, 
                                                        seed=args.seed, 
                                                        device=device, 
                                                        is_segmented=False)

        if not args.desegment:
            predictions = ' '.join(predictions)

        print('Translation: {}'.format(predictions))

    if args.translate_file:
        print('Loading source file')
        with open(args.src_file, 'r') as f:
            src_sentences = [l.strip() for l in f]
        
        print('Producing translations')
        translations = []
        for s in tqdm(src_sentences):
            predictions, _ = translate_one_sentence(s, 
                                                    model, 
                                                    args.spm_model, 
                                                    args.src_vocab, 
                                                    idx2word, 
                                                    word2idx, 
                                                    vocab_set, 
                                                    msl=args.msl, 
                                                    desegment=args.desegment, 
                                                    max_words=args.max_words, 
                                                    seed=args.seed, 
                                                    device=device, 
                                                    is_segmented=True)
            if not args.desegment: predictions = ' '.join(predictions)
            translations.append(predictions)
        
        print('Writing to file')
        with open(args.output_file, 'w') as f:
            for line in translations:
                f.write(line + '\n')
        print('Done!')

if __name__ == '__main__':
    main()
