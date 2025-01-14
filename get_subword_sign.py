from transformers import AutoTokenizer

def get_subword_sign(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab = tokenizer.get_vocab()
    
    # Check for common subword signs
    subword_signs = ["##", "Ġ", "▁"]
    for sign in subword_signs:
        for token in vocab:
            if sign in token:
                return sign
    return None
