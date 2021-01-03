import argparse

from sacremoses import MosesDetokenizer
import sacrebleu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--translation_file', type=str, help='File with translations (must be desegmented)')
    parser.add_argument('--reference_file', type=str, help='Reference file for target language')
    parser.add_argument('--tgt_lang', type=str, default='de', help='Target language code')

    args = parser.parse_args()

    # Detokenize the files
    detok = MosesDetokenizer(lang=args.tgt_lang)
    with open(args.translation_file, 'r') as f:
        translations = [detok.detokenize(l.strip().split()) for l in f]
    with open(args.reference_file, 'r') as f:
        refs = [detok.detokenize(l.strip().split()) for l in f]
    refs = [refs]

    # Compute BLEU score
    bleu = sacrebleu.corpus_bleu(translations, refs)
    print(bleu.score)

if __name__ == '__main__':
    main()
