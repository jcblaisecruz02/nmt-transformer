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

We use the WMT 14 German-English parallel corpus for training our Transformer model. A preprocessed version can be accessed through the Stanford NLP Group's website [here](https://nlp.stanford.edu/projects/nmt/). For testing, we use the Newstest 2014 parallel corpus for English and German. The test data can also be found in the same site.

Download the data as follows:
```
mkdir wmt14 && cd wmt14
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.en
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.de
```

First, we merge the two language data files, shuffle them, then train a joint BPE model (since the Transformer uses tied weights). This should give you a joint model and a joint vocabulary.

```
cat wmt14/train.en wmt14/train.de | shuf > wmt14/train.joint
spm_train --input=wmt14/train.joint --model_prefix=wmt14joint --vocab_size=37000 --character_coverage=1.0 --model_type=bpe --pad_id=3
mkdir tokenizers
mv wmt14joint.model tokenizers/wmt14joint.model && mv wmt14joint.vocab tokenizers/wmt14joint.vocab
```

We have to isolate subwords that are seen only in the target language from the source language and vice versa so that the encoder/decoder will not encounter them at train time, since it should be unknown. More information on this can be found in the [subword-nmt](https://github.com/rsennrich/subword-nmt#best-practice-advice-for-byte-pair-encoding-in-nmt) github. This will produce vocabulary files for the source and target languages, but constrained from the vocabulary of the joint BPE model.

```
spm_encode --model=tokenizers/wmt14joint.model --generate_vocabulary < wmt14/train.en > tokenizers/wmt14en.vocab
spm_encode --model=tokenizers/wmt14joint.model --generate_vocabulary < wmt14/train.de > tokenizers/wmt14de.vocab
```

Lastly, we segment the train and test files using the isolated vocabularies.

```
spm_encode --model=tokenizers/wmt14joint.model --vocabulary=tokenizers/wmt14en.vocab --extra_options=bos:eos --output_format=piece < wmt14/train.en > wmt14/train_tokenized.en
spm_encode --model=tokenizers/wmt14joint.model --vocabulary=tokenizers/wmt14de.vocab --extra_options=bos:eos --output_format=piece < wmt14/train.de > wmt14/train_tokenized.de
spm_encode --model=tokenizers/wmt14joint.model --vocabulary=tokenizers/wmt14en.vocab --extra_options=bos:eos --output_format=piece < wmt14/newstest2013.en > wmt14/test_tokenized.en
spm_encode --model=tokenizers/wmt14joint.model --vocabulary=tokenizers/wmt14de.vocab --extra_options=bos:eos --output_format=piece < wmt14/newstest2013.de > wmt14/test_tokenized.de
```

This entire process should take more or less an hour to finish on a sufficiently fast CPU.

After segmentation, we need to convert each source-target pair in the training and test datasets into their own `.txt` file for better RAM management later on. This can be done using the following provided script:

```
python nmt-transformer/preprocess.py \
    --src_train wmt14/train_tokenized.en \
    --trg_train wmt14/train_tokenized.de \
    --src_test wmt14/test_tokenized.en \
    --trg_test wmt14/test_tokenized.de \
    --path dataset
```

Once this is done, we can now proceed to training the model.

# Model Training

Training an NMT model is straightforward. This trains the smallest model (Model C, 30M parameters) reported on the Transformer paper for the WMT German-English translation task (N=2, d_model=512, pf_dim=2048, h=8, dropout=0.1):

```
python nmt-transformer/main.py \
    --save_dir transformer_base_modelc \
    --do_test \
    --do_train \
    --train_dir dataset/train \
    --valid_dir dataset/test \
    --test_dir dataset/test \
    --src_vocab tokenizers/wmt14joint.vocab \
    --trg_vocab tokenizers/wmt14joint.vocab \
    --num_workers 4 \
    --src_msl 100 \
    --trg_msl 100 \
    --tie_weights \
    --hidden_dim 512 \
    --n_layers 2 \
    --n_heads 8 \
    --pf_dim 2048 \
    --dropout 0.1 \
    --optimizer adamw \
    --learning_rate 4.4e-2 \
    --adam_epsilon 1e-9 \
    --adam_b1 0.9 \
    --adam_b2 0.98 \
    --weight_decay 0.0 \
    --scheduler linear \
    --warmup_pct 0.04 \
    --clip 1.0 \
    --batch_size 128 \
    --epochs 3 \
    --seed 1111
```
This uses the `AdamW` optimizer and `get_linear_schedule_with_warmup` scheduler from HuggingFace Transformers. The script will use `torch.optim.Adam` as its default optimizer, and will not use a scheduler by default. The script will save checkpoints in the directory passed to `--save_dir`. Training can be resumed from the checkpoint by using the `--resume_training` flag.

We use the same `--src_vocab` and `--trg_vocab` (`tokenizers/wmt14joint.vocab`) since we are using `--tie_weights` as directed in the paper. If you do not wish to tie the encoder/decoder embedding weights and the projection layer weights, you can opt to use language-specific BPE vocabularies instead (such as `tokenizers/wmt14en.vocab` and `tokenizers/wmt14de.vocab`). This, however, will result in a much larger model size and slower training time.

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
- [x] Added weight tying.
- [x] Added full instructions for training joint BPE.

**December 15, 2020**
- [x] Added initial training scripts.
- [x] Added support for FP16 training via Apex.
- [x] Switched dataloading to custom dataloaders to save RAM.
- [x] Added model checkpointing.

**December 13, 2020**
- [x] Initial commit.
- [x] Added preprocessing instructions for SentencePiece tokenizer training.
- [x] Tested maximum data chunk sizes that RAM and GPU can handle.

# Contributing
If you see any bugs or have any questions, do drop by the Issues tab! Contributions and pull requests are welcome as well.
