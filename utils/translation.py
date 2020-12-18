import torch 
import collections
import numpy as np
import math

from .data import segment, pad_and_truncate, unsegment

def translate(sample, model, idx2word, word2idx, max_words=20, seed=42, device=None):
    '''Input has to be a torch longtensor that's segmented, padded, and processed'''
    torch.manual_seed(seed)

    if type(model) is torch.optim.swa_utils.AveragedModel:
        model = model.module

    with torch.no_grad():
        src_mask = model.make_src_mask(sample).to(device)
        enc_src = model.encoder(sample, src_mask)

        tokens = [word2idx['<s>']]

        for _ in range(max_words):
            trg_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
            trg_mask = model.make_trg_mask(trg_tensor).to(device)
            out, attention = model.decoder(trg_tensor, enc_src, src_mask, trg_mask)

            pred_ix = out.argmax(2)[:, -1].item()
            tokens.append(pred_ix)    

            if pred_ix == word2idx['</s>']: break

    # Convert predictions from indices to text. Cut the attentions to translation length
    predictions = [idx2word[ix] for ix in tokens]
    attention = attention.squeeze(0).cpu().numpy()

    return predictions, attention

def translate_one_sentence(s, model, spm_model, src_vocab, idx2word, word2idx, vocab_set, msl=100, desegment=False, max_words=20, seed=42, device=None, is_segmented=False):
    if not is_segmented: s = segment(s, spm_model, src_vocab)
    input_ids = torch.LongTensor([pad_and_truncate(s, word2idx, msl, vocab_set)]).to(device)

    predictions, attention = translate(input_ids, model, idx2word, word2idx, max_words=max_words, seed=seed, device=device)
    
    if desegment: 
        predictions = ' '.join(predictions)
        predictions = unsegment(predictions, spm_model)

    return predictions, attention

def _get_ngrams(segment, max_order):
    """Extracts all n-grams up to a given maximum order from an input segment.
    Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned by this
                methods.
    Returns:
        The Counter containing all n-grams up to max_order in segment
        with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts

# BLEU Score implementation Copyright 2020 Tensor2Tensor authors

def compute_bleu(reference_corpus, translation_corpus, max_order=4, use_bp=True):
    """Computes BLEU score of translated segments against one or more references.
    Args:
        reference_corpus: list of references for each translation. Each
                reference should be tokenized into a list of tokens.
        translation_corpus: list of translations to score. Each translation
                should be tokenized into a list of tokens.
        max_order: Maximum n-gram order to use when computing BLEU score.
        use_bp: boolean, whether to apply brevity penalty.
    Returns:
        BLEU score.
    """
    reference_length = 0
    translation_length = 0
    bp = 1.0
    geo_mean = 0

    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    precisions = []

    for (references, translations) in zip(reference_corpus, translation_corpus):
        reference_length += len(references)
        translation_length += len(translations)
        ref_ngram_counts = _get_ngrams(references, max_order)
        translation_ngram_counts = _get_ngrams(translations, max_order)

        overlap = dict((ngram, min(count, translation_ngram_counts[ngram])) for ngram, count in ref_ngram_counts.items())

        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for ngram in translation_ngram_counts:
            possible_matches_by_order[len(ngram)-1] += translation_ngram_counts[ngram]
    precisions = [0] * max_order
    smooth = 1.0
    for i in range(0, max_order):
        if possible_matches_by_order[i] > 0:
            precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
            if matches_by_order[i] > 0:
                precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
            else:
                smooth *= 2
                precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
        else:
            precisions[i] = 0.0

    if max(precisions) > 0:
        p_log_sum = sum(math.log(p) for p in precisions if p)
        geo_mean = math.exp(p_log_sum/max_order)

    if use_bp:
        if not reference_length:
            bp = 1.0
        else:
            ratio = translation_length / reference_length
            if ratio <= 0.0:
                bp = 0.0
            elif ratio >= 1.0:
                bp = 1.0
            else:
                bp = math.exp(1 - 1. / ratio)
    bleu = geo_mean * bp
    return np.float32(bleu)
