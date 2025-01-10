import pickle
from BPE import get_stats, merge
import regex as re

# Load merges and vocab from the file
with open('bpe_results.pkl', 'rb') as f:
    merges, ids, num_merges = pickle.load(f)

# Define the GPT-2 regex pattern (same as in BPE.py)
gpt2pat = re.compile(r"""
    # Simpler syllable-based grouping
    (?:[\p{Devanagari}&&[क-ह]][ा-ौ\u093C\u0901-\u0903]?)  # Consonant + modifiers
    |[\u0905-\u0914]    # Independent vowels
    |[क-ह]्[क-ह]       # Basic conjuncts
    |\p{N}+            # Numbers
    |\s+               # Whitespace
    |[।॥]             # Punctuation
    |[^\s\p{Devanagari}\p{N}]+  # Other characters
    """, re.VERBOSE)

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
    # Debug printing
    print("Vocabulary contents:")
    for idx, byte_seq in vocab.items():
        try:
            char = byte_seq.decode('utf-8')
            print(f"ID {idx}: bytes {list(byte_seq)} -> '{char}'")
        except UnicodeDecodeError:
            print(f"ID {idx}: bytes {list(byte_seq)} -> [INVALID UTF-8]")
    
    print("\nDecoding sequence:")
    tokens = []
    for idx in ids:
        if idx in vocab:
            token_bytes = vocab[idx]
            try:
                char = token_bytes.decode('utf-8')
                print(f"ID {idx} -> '{char}'")
            except UnicodeDecodeError:
                print(f"ID {idx} -> [INVALID UTF-8] {list(token_bytes)}")
            tokens.append(token_bytes)
        else:
            print(f"Missing ID: {idx}")
    
    # Original decoding logic
    text = b''.join(tokens).decode('utf-8', errors='replace')
    
    # Write the decoded text to a new file
    with open('decoded_output.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    
    return text

# Example: Decode a list of IDs
set_of_ids = [262, 32, 32, 32, 32, 32, 32, 32, 32, 342, 32, 287, 281, 32, 32, 32, 266, 32, 32, 32, 32, 32, 32, 32, 32, 260, 32, 32, 32, 32, 32, 1719, 32, 32, 32, 46, 32, 32, 265, 32, 308, 32, 32, 317, 32, 32, 639, 32, 32, 32, 32, 32, 32, 44, 32, 272, 32, 265, 32, 32, 32, 611, 32, 410, 32, 32, 313, 32, 354, 32, 32, 32, 32, 46, 32, 32, 32, 32, 32, 32, 32, 32, 262, 32, 32, 32, 32, 32, 32, 262, 32, 32, 32, 267, 32, 297, 32, 32, 32, 32, 260, 32, 44, 32, 32, 32, 32, 32, 32, 265, 32, 32, 32, 32, 32, 32, 32, 267, 293, 32, 262, 32, 32, 32, 32, 46, 270, 666, 32, 396, 32, 262, 32, 32, 353, 829, 32, 32, 44, 32, 34, 32, 32, 32, 32, 32, 266, 32, 46, 32, 32, 32, 32, 32, 32, 32, 32, 314, 32, 32, 32, 32, 32, 32, 265, 32, 32, 32, 32, 32, 32, 46, 32, 32, 32, 32, 32, 32, 354, 32, 32, 260, 32, 32, 267, 293, 32, 32, 32, 32, 267, 293, 32, 32, 32, 32, 32, 32, 32, 46, 32, 265, 260, 32, 32, 32, 639, 32, 32, 32, 32, 32, 32, 32, 32, 260, 46, 32, 32, 32, 32, 32, 32, 32, 32, 32, 46, 32, 265, 32, 32, 32, 32, 32, 32, 32, 32, 32, 46, 32, 272, 32, 32, 262, 32, 32, 32, 32, 32, 32, 44, 32, 32, 32, 32, 32, 32, 45, 32, 32, 342, 287, 32, 32, 32, 260, 298, 32, 40, 32, 32, 351, 41, 32, 32, 32, 32, 32, 32, 32, 265, 260, 32, 267, 293, 32, 32, 32, 32, 260, 760]
decoded_text = decode(set_of_ids)  # Pass the list of IDs
print(decoded_text)

def encode():
    # Read input text from a new file
    with open('encode_input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Tokenize the text using the regex pattern
    tokens = re.findall(gpt2pat, text)
    
    # Convert tokens to byte sequences and maintain grouping
    byte_tokens = [token.encode('utf-8') for token in tokens]
    token_list = [list(token) for token in byte_tokens]
    
    # Process each token
    final_tokens = []
    for token in token_list:
        current_token = list(token)
        while len(current_token) >= 2:
            stats = get_stats([current_token])
            if not stats:
                break
            pair = min(stats, key=lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break
            idx = merges[pair]
            current_token = merge([current_token], pair, idx)[0]
        final_tokens.extend(current_token)
    
    return final_tokens

# Example: Encode text from a file
#encoded_tokens = encode()
#print(encoded_tokens)