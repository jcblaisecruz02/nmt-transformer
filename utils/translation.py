import torch 
from .data import segment, pad_and_truncate, unsegment

def translate(sample, model, idx2word, word2idx, max_words=20, seed=42, device=None):
    '''Input has to be a torch longtensor that's segmented, padded, and processed'''
    torch.manual_seed(seed)

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
