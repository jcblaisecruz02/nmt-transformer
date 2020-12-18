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

Training an NMT model is straightforward. This trains using the smallest model configuration (Model C, 30M parameters) reported on the Transformer paper for the WMT German-English translation task (N=2, d_model=512, pf_dim=2048, h=8, dropout=0.1):

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
    --learning_rate 1e-3 \
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

This setup should give you the following results:

```
Test Loss 2.0481 | Test Ppl 7.7532
```

This uses the `AdamW` optimizer and `get_linear_schedule_with_warmup` scheduler from HuggingFace Transformers. The script will use `torch.optim.Adam` as its default optimizer, and will not use a scheduler by default. The script will save checkpoints in the directory passed to `--save_dir`. Training can be resumed from the checkpoint by using the `--resume_training` flag.

We use the same `--src_vocab` and `--trg_vocab` (`tokenizers/wmt14joint.vocab`) since we are using `--tie_weights` as directed in the paper. If you do not wish to tie the encoder/decoder embedding weights and the projection layer weights, you can opt to use language-specific BPE vocabularies instead (such as `tokenizers/wmt14en.vocab` and `tokenizers/wmt14de.vocab`). This, however, will result in a much larger model size and slower training time.

For speedups, we suggest using NVIDIA Apex for 16-bit floating point training. Enabling this for the training script only requires adding the following flags:

```
    --fp16 \
    --opt_level O1
```

To use settings indicated in the paper, set the following flags:

```
    --scheduler noam \
    --warmup_steps 4000 \
    --criterion label_smoothing \
    --smoothing 0.1 \
    ...
```

The `--warmup_steps` parameter will override the `--warmup_pct` parameter. Make sure to compute how many batches you are training per epoch to use an appropriate number of warmup steps relative to the number of total steps.

For stochastic weight averaging, you may use the following parameters:

```
    --use_swa \
    --swa_pct 0.02 \
    --swa_times 5 \
    ...
```

This performs model averaging five times over the last 10% of total steps, with an interval of 2% of the total steps. Using the `--use_swa` flag will save the actual model (`model.bin`) and the averaged model (`swa_model.bin`) separately in the `--save_dir` directory.

For more information on reproduction scores and setups, see [Results and Reproduction Milestones](https://github.com/jcblaisecruz02/nmt-transformer#results-and-reproduction-milestones) below.

# Producing Translations
There are two translation modes: single sentence translation, and file translation.

To translate a single unsegmented, untokenized sentence from the source to the target language, the following command may be used:

```
python nmt-transformer/translate.py \
    --translate_sentence \
    --sentence "Republican leaders justified their policy by the need to combat electoral fraud ." \
    --src_vocab tokenizers/wmt14en.vocab \
    --joint_vocab tokenizers/wmt14joint.vocab \
    --spm_model tokenizers/wmt14joint.model \
    --save_dir transformer_base_modelc \
    --desegment \
    --msl 100 \
    --max_words 20 \
    --seed 42
```

This should give the following output:

```
Translation: Die republikanischen Führer haben ihre Politik durch die Notwendigkeit der Bekämpfung von Wahlbetrug gerechtfertigt .
```

The script takes care tokenization, segmentation, translation, and desegmentation. There is no need for preprocessing to translate a single sentence via the provided script.

We can also translate an entire file (say, for producing translations of the test set). This assumes that the input test file has already been segmented via SentencePiece. To produce translations this way, the following command may be used:

```
python nmt-transformer/translate.py \
    --translate_file \
    --src_file tokenized_wmt14/test_tokenized.en \
    --output_file output.txt \
    --src_vocab tokenizers/wmt14en.vocab \
    --joint_vocab tokenizers/wmt14joint.vocab \
    --spm_model tokenizers/wmt14joint.model \
    --save_dir transformer_base_modelc \
    --desegment \
    --msl 100 \
    --max_words 100 \
    --seed 42 \
    --use_cuda
```

This should produce a translation of the WMT14 Test set (newstest2013) in about 10 minutes. If your input file has not been segmented by SentencePiece yet, remove the `--desegment` toggle from the command line arguments (do note that this will increase the translation time by 3x). We highly encourage the use of `--use_cuda` during translation to speed up the process.

The translation script will use the non-averaged saved model by default. To use the averaged model, use the following flag:

```
    --use_swa \
    ...
```

To get a BLEU score for the translated corpus, use the following provided script:

```
python nmt-transformer/bleu.py \
    --translation_file output.txt \
    --reference_file wmt14/newstest2013.de
```

If you're using a translation from the checkpoint produced by the example above, you should get the following output:

```
BLEU: 20.08
```

# Results and Reproduction Milestones
*TBA*

# Changelog
**December 18, 2020**
- [x] Added Noam scheduling.
- [x] Added Label Smoothing.
- [x] Added Stochastic Weight Averaging (SWA).

**December 17, 2020**
- [x] Added translation scripts and modes.
- [x] Added support for auto de/segmentation in the utilities.
- [x] Added BLEU scoring.

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
