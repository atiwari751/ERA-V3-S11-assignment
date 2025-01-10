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
set_of_ids = [2532, 522, 258, 3103, 425, 332, 374, 2797, 44, 2391, 1508, 369, 63, 1375, 39, 261, 972, 277, 641, 385, 44, 2208, 553, 425, 1592, 63, 330, 39, 318, 1088, 285, 843, 405, 261, 46, 330, 39, 109, 1070, 325, 259, 888, 2913, 522, 1796, 524, 46, 966, 824, 306, 262, 354, 820, 726, 522, 2913, 1796, 524, 294, 330, 2827, 369, 44, 330, 824, 306, 262, 279, 551, 46, 966, 672, 2988, 306, 301, 3188, 451, 270, 814, 44, 330, 672, 1726, 285, 306, 1475, 46]
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