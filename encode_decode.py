import pickle
from BPE import get_stats, merge

# Load merges and vocab from the file
with open('bpe_results.pkl', 'rb') as f:
    merges, ids, num_merges = pickle.load(f)

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
    # given ids (list of integers), return Python string
    tokens = [vocab[idx].decode("utf-8", errors="replace") for idx in ids]
    text = '    '.join(tokens)  # Join tokens with a single space
    
    # Write the decoded text to a new file
    with open('decoded_output.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    
    return text

# Example: Decode a list of IDs
set_of_ids = [312, 1366, 565, 278, 302, 717, 256, 429, 1496, 1687, 808, 411, 110, 2862, 289, 670, 312, 1366, 39, 1281, 1191, 2358, 456, 374, 2453, 574, 429, 1687, 670, 73, 39, 353, 1176, 286, 904, 367, 279, 2310, 39, 695, 1398, 999, 806, 1271, 3455, 565, 119, 1902, 103, 2310, 116, 851, 403, 379, 260, 846, 2713, 565, 3466, 119, 114, 588, 292, 360, 1263, 258, 1285, 1402, 403, 3305, 114, 1278, 73, 116, 887, 773, 363, 403, 279, 2035, 274, 1150, 3273, 887, 2398, 1219, 1031, 2514, 46]
decoded_text = decode(set_of_ids)  # Pass the list of IDs
print(decoded_text)

def encode():
    # Read input text from a new file
    with open('encode_input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # given a string, return list of integers (the tokens)
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # nothing else can be merged
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    
    return tokens

# Example: Encode text from a file
encoded_tokens = encode()
print(encoded_tokens)