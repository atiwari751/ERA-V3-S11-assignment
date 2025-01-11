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
set_of_ids = [2342, 307, 295, 286, 1413, 302, 839, 644, 574, 982, 3877, 405, 1086, 272, 978, 181, 3927, 1171, 294, 274, 964, 438, 767, 337, 284, 361, 332, 286, 776, 315, 2331, 429, 841, 631, 385, 1694, 273, 310, 418, 1607, 445, 935, 286, 962, 1244, 698, 294, 3069, 347, 46, 450, 1462, 259, 646, 302, 554, 276, 2252, 334, 292, 2835, 2500, 315, 1006, 3367, 302, 296, 1299, 330, 289, 44, 327, 345, 1413, 286, 2911, 1906, 2592, 1322, 888, 330, 279, 711, 1474, 997, 1068, 295, 1236, 347, 46, 513, 1067, 579, 1194, 2596, 286, 847, 732, 307, 295, 309, 1423, 1953, 340, 555, 563, 1413, 286, 376, 466, 596, 294, 315, 385, 347, 44, 1001, 478, 776, 1068, 295, 1236, 919, 1216, 315, 345, 1115, 315, 3189, 481, 437, 340, 557, 1125, 1135, 1501, 857, 289, 46, 10, 10, 53, 53, 2794, 732, 307, 295, 317, 2705, 2246, 280, 1308, 698, 486, 309, 739, 44, 32, 34, 808, 830, 1015, 516, 1315, 544, 667, 289, 46, 513, 776, 1914, 311, 286, 948, 294, 856, 915, 2438, 658, 367, 271, 272, 564, 516, 472, 340, 1571, 1423, 2592, 286, 638, 416, 1953, 46, 586, 462, 1315, 544, 3075, 583, 888, 330, 588, 444, 557, 1448, 739, 340, 737, 1068, 295, 1236, 919, 1216, 294, 3253, 776, 391, 1410, 46, 1496, 1448, 292, 2835, 2500, 294, 738, 1374, 3075, 583, 330, 2660, 3252, 904, 46, 1441, 315, 1448, 1374, 3075, 583, 330, 1473, 481, 437, 46, 345, 778, 1758, 1307, 315, 2210, 3075, 583, 299, 333, 751, 259, 420, 46, 327, 766, 1200, 294, 1448, 499, 1394, 739, 437, 44, 707, 450, 413, 340, 3602, 1135, 45, 864, 261, 2660, 2749, 1930, 286, 847, 447, 1782, 1633, 510, 308, 306, 583, 399, 1508, 2632, 261, 41, 309, 462, 1135, 330, 391, 1193, 1496, 557, 1574, 776, 3189, 1340, 3435]
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