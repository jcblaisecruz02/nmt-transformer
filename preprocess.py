import os
import argparse
from tqdm import tqdm

def write_files(path, src, trg):
    with open(src, 'r') as f:
        src_text = [l.strip() for l in f]
    with open(trg, 'r') as f:
        trg_text = [l.strip() for l in f]
        
    os.mkdir(path)
    for i in tqdm(range(0, len(src_text))):
        with open(path + '/{}.txt'.format(i), 'w') as f:
            f.write(src_text[i] + '\n' + trg_text[i] + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_train', type=str)
    parser.add_argument('--trg_train', type=str)
    parser.add_argument('--src_valid', type=str)
    parser.add_argument('--trg_valid', type=str)
    parser.add_argument('--src_test', type=str)
    parser.add_argument('--trg_test', type=str)
    parser.add_argument('--path', type=str, required=True)
    
    args = parser.parse_args()
    print(args)
    
    os.mkdir(args.path)
    if args.src_train is not None and args.trg_train is not None:
        print("Writing train files.")
        write_files(args.path + '/train', args.src_train, args.trg_train)
    if args.src_valid is not None and args.trg_valid is not None:
        print("Writing validation files.")
        write_files(args.path + '/valid', args.src_valid, args.trg_valid)
    if args.src_test is not None and args.trg_test is not None:
        print("Writing test files.")
        write_files(args.path + '/test', args.src_test, args.trg_test)

if __name__ == '__main__':
    main()
