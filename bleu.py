import argparse

from utils.translation import compute_bleu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--translation_file', type=str, help='File with translations (must be desegmented)')
    parser.add_argument('--reference_file', type=str, help='Reference file for target language')

    args = parser.parse_args()

    with open(args.reference_file, 'r') as f:
        references = [l.strip().split(' ') for l in f]
    with open(args.translation_file, 'r') as f:
        translations = [l.strip().split(' ') for l in f]

    bleu = compute_bleu(references, translations)
    print("BLEU: {:.2f}".format(bleu * 100))

if __name__ == '__main__':
    main()
