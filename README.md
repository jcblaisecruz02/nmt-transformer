# MNT Transformer

# Requirements

# Data Processing

We use the WMT 14 German-English parallel corpus for training our Transformer model. A preprocessed version can be accessed through the Stanford NLP Group's webiste [here](https://nlp.stanford.edu/projects/nmt/). For testing, we use the Newstest 2014 parallel corpus for English and German. The test data can also be found in the same site.

Download the data as follows:
```
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.en
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.de
```

We build [SentencePiece](https://github.com/google/sentencepiece) tokenizers using our training corpus as follows:

```
spm_train --input=train.en --model_prefix=wmt14en --vocab_size=32000 --character_coverage=1.0 --model_type=bpe --pad_id=3
spm_train --input=train.de --model_prefix=wmt14de --vocab_size=32000 --character_coverage=1.0 --model_type=bpe --pad_id=3
```

Encode the input files using the generated sentencepiece models:

```
spm_encode --model=wmt14en.model --extra_options=bos:eos --output_format=piece < train.en > train_tokenized.en
spm_encode --model=wmt14de.model --extra_options=bos:eos --output_format=piece < train.de > train_tokenized.de
spm_encode --model=wmt14en.model --extra_options=bos:eos --output_format=piece < newstest2013.en > test_tokenized.en
spm_encode --model=wmt14de.model --extra_options=bos:eos --output_format=piece < newstest2013.de > test_tokenized.de
```

Encoding takes about 18 minutes to fully finish.

Next, we need to stream the dataset, converting each line of example into its own file. We do this to lazily load the data to save space.

```
python nmt-transformer/preprocess.py \
    --src_test dataset/test_tokenized.en \
    --trg_test dataset/test_tokenized.de \
    --path data
```

# Model Training

```
python nmt-transformer/main.py \
    --do_test \
    --test_dir data/test \
    --src_vocab tokenizers/wmt14en.vocab \
    --trg_vocab tokenizers/wmt14de.vocab \
    --src_msl 45 \
    --trg_msl 55 \
    --batch_size=128 \
    --num_workers=1
```