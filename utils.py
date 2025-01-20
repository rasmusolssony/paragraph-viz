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

def get_start_token(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab = tokenizer.get_vocab()
    
    start_token = None
    start_tokens = ["<s>", "<sos>", "[CLS]"]
    for token in vocab:
        for start in start_tokens:
            if start in token:
                start_token = token
                break
            
    if start_token is None:
        print("Start token not found")

    return start_token

def get_end_token(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab = tokenizer.get_vocab()

    end_token = None
    end_tokens = ["</s>", "<eos>", "[SEP]", "[EOS]"]
    for token in vocab:
        for end in end_tokens:
            if end in token:
                end_token = token
                break
    
    if end_token is None:
        print("End token not found")

    return end_token