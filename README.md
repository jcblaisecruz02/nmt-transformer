# NMT Transformer
This repository is my attempt at a reproduction of the original [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper by Vaswani et al., 2017. Scripts are provided to perform end-to-end preprocessing, training, inference translation, BLEU scoring, and attention visualization.

Currently, this is a work in progress, so expect sharp edges and unimplemented portions of the paper. Please check the changelog below for more information on the current state of the project.

# Requirements
* [SentencePiece](https://github.com/google/sentencepiece) for training tokenizers.
* PyTorch v1.x.
* NVIDIA Apex for FP16 training.
* NVIDIA GPU (all experiments were run on NVIDIA Tesla V100 GPUs).
* [PyTorch LAMB](https://github.com/cybertronai/pytorch-lamb) (Optional) for LAMB optimizer.
* [HuggingFace Transformers](https://huggingface.co/transformers/index.html) (Optional) for AdamW and linear warmup scheduler.

# Data Processing

We use the WMT 14 German-English parallel corpus for training our Transformer model. A preprocessed version can be accessed through the Stanford NLP Group's webiste [here](https://nlp.stanford.edu/projects/nmt/). For testing, we use the Newstest 2014 parallel corpus for English and German. The test data can also be found in the same site.

Download the data as follows:
```
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.en
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.de
```

We build SentencePiece tokenizers using our training corpus as follows:

```
spm_train --input=train.en --model_prefix=wmt14en --vocab_size=32000 --character_coverage=1.0 --model_type=bpe --pad_id=3
spm_train --input=train.de --model_prefix=wmt14de --vocab_size=32000 --character_coverage=1.0 --model_type=bpe --pad_id=3

mkdir tokenizers
mv wmt14en.model tokenizers/wmt14en.model && mv wmt14en.vocab tokenizers/wmt14en.vocab
mv wmt14de.model tokenizers/wmt14de.model && mv wmt14de.vocab tokenizers/wmt14de.vocab
```

Encode the input files using the generated sentencepiece models:

```
mkdir data
spm_encode --model=wmt14en.model --extra_options=bos:eos --output_format=piece < train.en > data/train_tokenized.en
spm_encode --model=wmt14de.model --extra_options=bos:eos --output_format=piece < train.de > data/train_tokenized.de
spm_encode --model=wmt14en.model --extra_options=bos:eos --output_format=piece < newstest2013.en > data/test_tokenized.en
spm_encode --model=wmt14de.model --extra_options=bos:eos --output_format=piece < newstest2013.de > data/test_tokenized.de
```

Encoding takes about 18 minutes to fully finish.

Next, we need to stream the dataset, converting each line of example into its own file. We do this to lazily load the data via PyTorch dataloaders later to save space.

```
python nmt-transformer/preprocess.py \
    --src_train data/train_tokenized.en \
    --trg_train data/train_tokenized.de \
    --src_test data/test_tokenized.en \
    --trg_test data/test_tokenized.de \
    --path dataset
```

After this is done, we can now begin training.

# Model Training

To train a Transformer MT model, we use the following command:

```
python nmt-transformer/main.py \
    --save_dir trained_model \
    --do_test \
    --do_train \
    --train_dir dataset/train \
    --valid_dir dataset/test \
    --test_dir dataset/test \
    --src_vocab tokenizers/wmt14en.vocab \
    --trg_vocab tokenizers/wmt14de.vocab \
    --num_workers 4 \
    --src_msl 100 \
    --trg_msl 100 \
    --hidden_dim 256 \
    --n_layers 3 \
    --n_heads 8 \
    --pf_dim 512 \
    --dropout 0.1 \
    --optimizer adamw \
    --learning_rate 1e-2 \
    --weight_decay 0.0 \
    --scheduler linear \
    --warmup_pct 0.1 \
    --clip 1.0 \
    --batch_size 128 \
    --epochs 10 \
    --seed 1111
```
This uses the `AdamW` optimizer and `get_linear_schedule_with_warmup` scheduler from HuggingFace Transformers. The script will use `torch.optim.Adam` as its default optimizer, and will not use a scheduler by default. The script will save checkpoints in the directory passed to `--save_dir`. Training can be resumed from the checkpoint by using the `--resume_training` flag.

For speedups, we suggest using NVIDIA Apex for 16-bit floating point training. Enabling this for the training script only requires adding the following flags:

```
    --fp16 \
    --opt_level O1
```

# Results and Reproduction Milestones
*TBA*

# Changelog
**December 16, 2020**
- [x] Added `AdamW` and `LAMB` optimizers.
- [x] Added linear warmup scheduler.
- [x] Added full support for checkpointing and training resuming.
- [x] Refactored code for training loops.
- [x] Added diagnostics reporting during training.

**December 15, 2020**
- [x] Added initial training scripts
- [x] Added support for FP16 training via Apex
- [x] Switched dataloading to custom dataloaders to save RAM
- [x] Added model checkpointing

**December 13, 2020**
- [x] Initial commit.
- [x] Added preprocessing instructions for SentencePiece tokenizer training.
- [x] Tested maximum data chunk sizes that RAM and GPU can handle.

# Contributing
If you see any bugs or have any questions, do drop by the Issues tab! Contributions and pull requests are welcome as well.
