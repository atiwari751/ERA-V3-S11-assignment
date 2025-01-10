import pickle
from BPE import get_stats, merge
import regex as re

# Load merges and vocab from the file
with open('bpe_results.pkl', 'rb') as f:
    merges, ids, num_merges = pickle.load(f)

# Define the GPT-2 regex pattern (same as in BPE.py)
gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
    # given ids (list of integers), return Python string
    tokens = [vocab[idx] for idx in ids]
    # Decode each token separately and join with tabs
    decoded_tokens = [token.decode("utf-8", errors="replace") for token in tokens]
    text = '\t'.join(decoded_tokens)
    
    # Write the decoded text to a new file
    with open('decoded_output.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    
    return text

# Example: Decode a list of IDs
set_of_ids = [1072, 342, 259, 1406, 338, 409, 332, 330, 545, 733, 44, 332, 110, 642, 336, 63, 1044, 263, 39, 263, 1496, 285, 97, 112, 410, 44, 1045, 421, 409, 915, 63, 301, 39, 299, 288, 305, 290, 496, 578, 263, 46, 301, 39, 109, 753, 619, 312, 261, 292, 879, 312, 342, 262, 114, 546, 46, 301, 116, 415, 308, 261, 329, 538, 1129, 342, 261, 292, 879, 312, 262, 114, 546, 304, 301, 1238, 115, 336, 44, 301, 415, 308, 261, 274, 567, 46, 301, 116, 373, 1179, 450, 308, 348, 118, 362, 119, 310, 500, 44, 301, 373, 1282, 290, 308, 282, 271, 1167, 46]
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
encoded_tokens = encode()
print(encoded_tokens)