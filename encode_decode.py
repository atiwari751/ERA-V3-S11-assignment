import pickle
from BPE import get_stats, merge
import regex as re

# Load merges and vocab from the file
with open('bpe_results.pkl', 'rb') as f:
    merges, ids, num_merges = pickle.load(f)

# Define the GPT-2 regex pattern (same as in BPE.py)
gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{N}+| ?(?:[\u0904-\u0939\u093d-\u093d\u0950-\u0950\u0958-\u0961\u0970-\u097f\ua8f2-\ua8fe\U00011b00-\U00011b09\u1cd3-\u1cd3\u1ce9-\u1cec\u1cee-\u1cf3\u1cf5-\u1cf6\u1cfa-\u1cfa][\u0900-\u0903\u093a-\u093c\u093e-\u094f\u0951-\u0957\u0962-\u0963\ua8e0-\ua8f1\ua8ff-\ua8ff\u1cd0-\u1cd2\u1cd4-\u1ce8\u1ced-\u1ced\u1cf4-\u1cf4\u1cf7-\u1cf9]*)+| ?\p{L}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
    # given ids (list of integers), return Python string
    tokens = [vocab[idx].decode("utf-8", errors="replace") for idx in ids]
    text = '\t'.join(tokens)  # Join tokens with tabs
    
    # Write the decoded text to a new file
    with open('decoded_output.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    
    return text

# Example: Decode a list of IDs
set_of_ids = [335, 332, 295, 401, 1050, 273, 1153, 1094, 294, 843,
859, 1092, 3583, 3327, 315, 2457, 437, 585, 867, 3747,
587, 299, 294, 315, 388, 3747, 587, 785, 414, 44,
1110, 712, 307, 295, 334, 984, 414, 329, 2892, 3747,
587, 583, 1160, 1593, 427, 3934, 621, 285, 1583, 1936,
294, 414, 260, 46, 548, 2007, 294, 2733, 294, 1467,
1553, 300, 763, 2045, 381, 285, 2093, 3934, 621, 1882,
315, 1077, 48, 44, 1581, 3991, 285, 1909, 315, 1595,
585, 46, 2161, 2714, 280, 1016, 698, 475, 316, 984,
45, 861, 261, 2836, 2999, 1947, 418, 329, 279, 3331,
266, 300, 44, 343, 591, 867, 3747, 587, 299, 330,
2457, 437, 585, 715, 57, 55, 1092, 3017, 294, 315,
565, 315, 565, 1467, 55, 489, 2139, 2057, 2927, 46,
54, 1553, 41, 2217, 2695, 315, 2457, 437, 585, 533,
46]
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
    
    # Calculate total bytes before compression
    total_bytes_before = sum(len(token) for token in token_list)
    
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
    
    # Calculate compression ratio
    compression_ratio = total_bytes_before / len(final_tokens)
    print(f"Compression ratio: {compression_ratio:.2f}X")
    
    return final_tokens, compression_ratio

# Example: Encode text from a file
encoded_tokens, ratio = encode()
print(f"Encoded tokens: {encoded_tokens}")