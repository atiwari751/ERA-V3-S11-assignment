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
set_of_ids = [34, 293, 474, 298, 275, 575, 1271, 260, 778, 1298, 763, 611, 1921, 310, 424, 352, 156, 347, 318, 947, 1410, 1832, 276, 2984, 314, 262, 770, 639, 2516, 1020, 3054, 260, 795, 1072, 993, 2392, 499, 474, 298, 275, 575, 611, 1921, 310, 424, 1271, 854, 940, 1761, 3036, 310, 424, 932, 1060, 661, 918, 342, 352, 156, 347, 318, 947, 1453, 1483, 324, 181, 347, 863, 591, 412, 606, 2234, 789, 481, 751, 587, 2039, 1750, 289, 301, 565, 278, 2675, 1532, 499, 1898, 820, 474, 298, 275, 575, 410, 3428, 1195, 569, 295, 3036, 310, 424, 352, 156, 347, 318, 947, 260, 1033, 2697, 495, 1832, 276, 2984, 761, 185, 1020, 3054, 377, 401, 430, 471, 953, 2232, 170, 474, 298, 275, 575, 611, 1921, 310, 424, 419, 2029, 185, 869, 268, 1254, 1998, 1842, 317, 2214, 1630, 376, 1141, 1709, 1909, 1842, 1200, 514, 171, 281, 798, 904, 510, 1865, 418, 264, 890, 1877, 272, 1254, 485, 1069, 967, 2100, 1046, 55, 2355, 2555, 563, 3352, 654, 1477, 1622, 708, 405, 1630, 859, 310, 1130, 289, 2817, 762, 336, 979, 1909, 867, 289, 3316, 1205, 1811, 1266, 2718, 886, 318, 1303, 1861, 1029, 44, 2697, 2033, 390, 606, 686, 143, 311, 1537, 271, 806, 286, 154, 1106, 563, 2460, 481, 786, 1061, 3036, 310, 424, 810, 2250, 881, 272, 1254, 1135]
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
#encoded_tokens = encode()
#print(encoded_tokens)