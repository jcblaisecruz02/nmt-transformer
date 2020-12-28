# NMT Transformer
This repository is my attempt at a reproduction of the original [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper by Vaswani et al., 2017. Scripts are provided to perform end-to-end preprocessing, training, inference translation, BLEU scoring, and attention visualization. 

My goal is to be able to reproduce ther results from the paper with just one GPU (the authors used 8 P100 GPUs). Due to the difference in the largest batch size fittable in one GPU, certain hyperparameter choices such as learning rate and training steps may differ. However, model sizes, dimensions, and architectures are made sure to remain the same. For more information on differences between implementations, see [Results and Reproduction Milestones](https://github.com/jcblaisecruz02/nmt-transformer#results-and-reproduction-milestones) below.

Currently, this is a work in progress, so expect sharp edges and unimplemented portions of the paper. Please check the changelog below for more information on the current state of the project.

# Table of Contents
* [Requirements](https://github.com/jcblaisecruz02/nmt-transformer#requirements)
* [Data Processing](https://github.com/jcblaisecruz02/nmt-transformer#data-processing)
* [Model Training](https://github.com/jcblaisecruz02/nmt-transformer#model-training)
* [Producing Translations](https://github.com/jcblaisecruz02/nmt-transformer#producing-translations)
* [Results and Reproduction Milestones](https://github.com/jcblaisecruz02/nmt-transformer#results-and-reproduction-milestones)
* [Changelog](https://github.com/jcblaisecruz02/nmt-transformer#changelog)
* [Contributing](https://github.com/jcblaisecruz02/nmt-transformer#contributing)

# Requirements
* [SentencePiece](https://github.com/google/sentencepiece) for training tokenizers.
* PyTorch v1.x.
* NVIDIA Apex for FP16 training.
* NVIDIA GPU (all experiments were run on NVIDIA Tesla V100 GPUs).

# Data Processing

I use the WMT 14 German-English parallel corpus for training our Transformer model. A preprocessed (pretokenized) version can be accessed through the Stanford NLP Group's website [here](https://nlp.stanford.edu/projects/nmt/). For testing, I use the Newstest 2013 parallel corpus for English and German. The test data can also be found in the same site.

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

Training an NMT model is straightforward. This setup reproduces the smallest model configuration (Model C, 30M parameters) reported on the Transformer paper for the WMT German-English translation task (N=2, d_model=512, pf_dim=2048, h=8, dropout=0.1):

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
    --optimizer adam \
    --learning_rate 1e-3 \
    --adam_epsilon 1e-9 \
    --adam_b1 0.9 \
    --adam_b2 0.98 \
    --weight_decay 0.0 \
    --scheduler noam \
    --warmup_steps 4190 \
    --criterion label_smoothing \
    --smoothing 0.1 \
    --clip 1.0 \
    --batch_size 128 \
    --epochs 3 \
    --seed 1111 \
    --use_swa \
    --swa_pct 0.05 \
    --swa_times 5
```

This setup should give you the following results:

```
Single   model loss: Test Loss 1.9404 | Test Ppl 6.9618
Averaged model loss: Test Loss 1.9296 | Test Ppl 6.8867
```

Our averaged model test perplexity is only 0.78 perplexity points lower than the one reported in the paper (6.11 Ppl). 

The script will save checkpoints in the directory passed to `--save_dir`. Training can be resumed from the checkpoint by using the `--resume_training` flag. 

To perform experiments with varying model settings, the following arguments may be toggled:
* `--tie_weights` — will tie the weights of the encoder embeddings, the decoder embeddings, and the softmax projection layer. When weights are tied, a joint vocabulary must be given to both the encoder and decoder. Otherwise, you can opt to use language-specific BPE vocabularies instead. This will, however, result in a much larger model size and training time.
* `--criterion` — the authors of the paper use label smoothing, which is implemented as `--criterion label_smoothing` in the training script. To use standard cross entropy loss, `cross_entropy` should be passed instead.
* `--smoothing` — controls label smoothing.
* `--optimizer` — the standard `torch.optim.Adam` implementation is used for training. You may opt to use weight-decay adjusted Adam from HuggingFace Transformers should you wish (you need to install the library). LAMB may also be used, provided PyTorch LAMB is installed.
* `--scheduler` — the script follow the paper's "noam schedule" for learning rates. If a scheduler is not indicated, the script will not use a scheduler by default. You may also opt to use a linear decay scheduler from HuggingFace Transformers for experiments.
* `--warmup_steps` and `--warmup_pct` — warmup steps is used for the original paper, and will override any passed arguments to warmup percentage.
* `--use_swa` — togges the use of stochastic weight averaging. Make sure to indicate `--swa_pct` and `--swa_times` as well. In the above example, this setting performs model averaging five times over the last 10% of total steps, with an interval of 2% of the total steps. Using the `--use_swa` flag will save the actual model (`model.bin`) and the averaged model (`swa_model.bin`) separately in the `--save_dir` directory.

For speedups, 16-bit floating point training is also available through NVIDIA Apex. Enabling this for the training script only requires adding the following flags:

```
    --fp16 \
    --opt_level O1
```
For more information on reproduction scores and setups, see [Results and Reproduction Milestones](https://github.com/jcblaisecruz02/nmt-transformer#results-and-reproduction-milestones) below.

# Producing Translations
To translate a single unsegmented, untokenized sentence from the source to the target language, the following command may be used:

```
python nmt-transformer/translate.py \
    --translate_sentence \
    --sentence "Republican leaders justified their policy by the need to combat electoral fraud ." \
    --joint_model tokenizers/wmt14joint.model \
    --joint_vocab tokenizers/wmt14joint.vocab \
    --src_vocab tokenizers/wmt14en.vocab \
    --save_dir base_modelc \
    --desegment \
    --beams 1 \
    --max_words 50 \
    --msl 100 \
    --seed 42
```

Using the model trained in the previous section should give the following output:

```
Die republikanischen Führer haben ihre Politik durch die Notwendigkeit , Wahlbetrug zu bekämpfen .
```

By default, the translation script will use greedy decoding (`--beams 1`). To use beam search with a higher beam width N, use `--beams N`. Do note that this will decode N^max_words - 1 tokens (with beam size 2 and max words 30, this is already 1 billion tokens) which is very computationally slow!

To use Top-K/Nucleus sampling, add the following flags:

```
    --use_topk \
    --top_k 50 \
    --top_p 0.92 \
    --temperature 0.3 \
```

Adding this to the earlier examples will give us the following translation:

```
Die republikanischen Führer haben ihre Politik durch die Notwendigkeit zur Bekämpfung von Wahlbetrug gerechtfertigt .
```


We can also translate an entire file (say, for producing translations of the test set). This assumes that the input test file has already been segmented via SentencePiece. To produce translations this way, the following command may be used:

```
python nmt-transformer/translate.py \
    --translate_file \
    --is_segmented \
    --src_file tokenized_wmt14/test_tokenized.en \
    --output_file output.txt \
    --joint_model tokenizers/wmt14joint.model \
    --joint_vocab tokenizers/wmt14joint.vocab \
    --src_vocab tokenizers/wmt14en.vocab \
    --save_dir base_modelc \
    --beams 1 \
    --use_topk \
    --top_k 50 \
    --top_p 0.92 \
    --temperature 0.3 \
    --max_words 50 \
    --msl 100 \
    --seed 42
```

This should produce a translation of the WMT14 Test set (newstest2013) in about 10 minutes. If your input file has not been segmented by SentencePiece yet, remove the `--desegment` toggle from the command line arguments (do note that this will increase the translation time by 3x). I highly encourage the use of `--use_cuda` during translation to speed up the process.

The translation script will use the non-averaged saved model by default. To use the averaged model, use the following flag:

```
python nmt-transformer/translate.py \
    --translate_file \
    ...
    --use_swa
```

To get a BLEU score for the tokenized translated corpus, use the following provided script:

```
python nmt-transformer/bleu.py \
    --translation_file output.txt \
    --reference_file tokenized_wmt14/test_tokenized.de
```

Using translations produced form the model trained in the previous section should give the following score:

```
BLEU: 24.6
```

This is 0.9 BLEU points higher than the reported test BLEU for the same model in the paper (23.7 BLEU).

# Results and Reproduction Milestones

Since our goal is to reproduce the results with just one GPU instead of the eight that the authors used, some considerations for the hyperparameters will be made:
* We use a slower learning rate due to the fact that we have a smaller maximum batch size compared to the authors' setup. A larger batch size can approximate the loss surface better, and thus can use a faster learning rate. Smaller batch sizes induce more noise than larger ones, which is why we use a slower rate.
* Our model averaging is approximate. The authors mention that for the base models, they average on the last five checkpoints during training, with each checkpoint written in 10-minute intervals. It is unrealistic for us to try and estimate which steps wrt. batch size these 10-minute intervals are, which is hard for reproduction. We instead opt to average N times over the last couple checkpoints written `--swa_pct` apart from each other.
* We also experiment with the number of steps needed to train the model. As with the learning rate problem, we may need more epochs to converge to a close solution as opposed to the authors' setup. The same can be said for the number of warmup steps. To be as close as possible, we set the number of warmup steps to be 4% of the total training steps by default.

While these are not *true replications* (as that would entail using the same hardware setup as well), I made sure to keep the actual model architecture the same even if the training hyperparameters are not.

Here is a table describing the current reproduction scores and progress:

| Model | Hyperparameters                 | PPL Reproduced     | PPL Difference       | BLEU Reproduced    | BLEU Difference       | Remarks            |
|-------|---------------------------------|--------------------|----------------------|--------------------|-----------------------|--------------------|
| Base  | `N=6`, `d_model=512`, `dff=2014`, `h=8` |  |  |  |  | In progress |
| C     | `N=2`        | :white_check_mark: 6.89 PPL| -0.78 | :white_check_mark: 24.6 BLEU | +0.9 | BLEU results using nucleus sampling. Uses positional embeddings. |
| E     | Positional embeddings instead of sinusoids |  |  |  |  | In progress |
| Notes | *\*Uses the same settings as the base model unless specified* | | *\*Lower PPL is better* | | *\*Higher BLEU is better* | |

To reproduce the results in the table, the commands used to train and translate the models are indicated below.

**Model C**
```
python nmt-transformer/main.py --save_dir transformer_base_modelc --do_test --do_train --train_dir dataset/train --valid_dir dataset/test --test_dir dataset/test --src_vocab tokenizers/wmt14joint.vocab --trg_vocab tokenizers/wmt14joint.vocab --num_workers 4 --src_msl 100 --trg_msl 100 --tie_weights --hidden_dim 512 --n_layers 2 --n_heads 8 --pf_dim 2048 --dropout 0.1 --optimizer adam --learning_rate 1e-3 --adam_epsilon 1e-9 --adam_b1 0.9 --adam_b2 0.98 --weight_decay 0.0 --scheduler noam --warmup_steps 4190 --criterion label_smoothing --smoothing 0.1 --clip 1.0 --batch_size 128 --epochs 3 --seed 1111 --use_swa --swa_pct 0.05 --swa_times 5
python nmt-transformer/translate.py --translate_file --is_segmented --src_file tokenized_wmt14/test_tokenized.en --output_file output.txt --joint_model tokenizers/wmt14joint.model --joint_vocab tokenizers/wmt14joint.vocab --src_vocab tokenizers/wmt14en.vocab --save_dir base_modelc --beams 1 --use_topk --top_k 50 --top_p 0.92 --temperature 0.3 --max_words 50 --msl 100 --seed 42
python nmt-transformer/bleu.py --translation_file output.txt --reference_file tokenized_wmt14/test_tokenized.de
```

# Changelog
**December 28, 2020**
- [x] Added Top-K and Nucleus Sampling.
- [x] Added updated BLEU results.

**December 25, 2020**
- [x] Added beam search.
- [x] Added initial reproduction results.

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
