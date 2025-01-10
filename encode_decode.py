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
set_of_ids = [297, 562, 373, 322, 310, 1454, 609, 1518, 405, 908, 261, 463, 110, 974, 316, 466, 297, 562, 39, 689, 1953, 112, 563, 268, 352, 1494, 1023, 587, 1509, 466, 73, 39, 307, 484, 1166, 525, 398, 315, 314, 73, 39, 347, 435, 815, 1421, 115, 284, 101, 257, 305, 373, 119, 114, 1061, 314, 73, 116, 436, 418, 465, 262, 712, 273, 562, 373, 115, 284, 101, 1316, 119, 114, 791, 295, 296, 1447, 259, 732, 595, 418, 115, 278, 114, 1333, 73, 116, 447, 1982, 418, 315, 717, 101, 119, 330, 517, 261, 73, 447, 119, 741, 304, 418, 1109, 116, 464, 46]
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