import torch 
import collections
import numpy as np
import math

from .data import segment, pad_and_truncate, unsegment

class State:
    def __init__(self, string, prob):
        self.string = string
        self.prob = prob

    def __str__(self):
        return "<{}: {}>".format(self.string, self.prob)

    def __eq__(self,  other):
        return str(self) == str(other)

def build_tree(model, state, beams, queue, candidates, max_words=30, early_stop=None, device=None):
    # If EOS or MSL, add the string to candidates
    if state.string[-1] == word2idx['</s>'] or len(state.string) > max_words:
        candidates.append((state.prob, state.string))

    # Else, use the current string to decode
    else:
        trg_tensor = torch.LongTensor(state.string).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor).to(device)
        with torch.no_grad():
            out, _ = model.decoder(trg_tensor, enc_src, src_mask, trg_mask)
            probs, ixs = torch.softmax(out[:, -1], dim=-1).cpu().topk(k=beams, dim=1)
        
        # Generate next states
        for i in range(beams): 
            new_string = state.string + [ixs[0, i].item()]
            new_prob = np.exp(np.log(state.prob) + np.log(probs[0, i].item()))
            new_child = State(new_string, new_prob)

            queue.append(new_child)

def beam_search(input_ids, model, idx2word, word2idx, beams=1, max_words=20, seed=42, device=None, return_top=1, strategy='bfs'):
    # Produce encoder outputs
    with torch.no_grad():
        src_mask = model.make_src_mask(input_ids).to(device)
        enc_src = model.encoder(input_ids, src_mask)
    
    # Initialize queue and root node
    queue = [State([word2idx['<s>']], 1)]
    candidates = []
    best = 0.0

    # Beam search
    i = 0
    while len(queue) > 0:
        state = queue.pop(0) if strategy == 'bfs' else queue.pop()
        build_tree(model, state, beams, queue, candidates, max_words, None, device)
        i += 1
        print(len(queue), end=' ')

    # Return the top n most probable candidates
    preds = sorted(candidates, key=lambda x: x[0], reverse=True)[:return_top]
    return [[idx2word[ix] for ix in p[1]] for p in preds]

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
