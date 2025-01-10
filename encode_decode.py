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
set_of_ids = [577, 284, 318, 304, 286, 3417, 1164, 477, 305, 297, 558, 1016, 446, 321, 1994, 1260, 1543, 2237, 181, 952, 273, 1884, 1593, 185, 301, 438, 272, 1027, 328, 284, 486, 321, 2902, 376, 468, 350, 685, 156, 1577, 268, 368, 976, 289, 332, 663, 172, 334, 2015, 646, 157, 1319, 732, 319, 389, 277, 1431, 508, 2842, 565, 579, 519, 144, 277, 2470, 323, 327, 337, 540, 2630, 376, 170, 283, 1151, 2183, 488, 1426, 285, 44, 388, 326, 810, 1087, 1428, 3017, 2989, 2328, 1196, 262, 1126, 928, 419, 170, 616, 645, 1097, 2328, 1732, 331, 267, 295, 726, 3446, 1751, 355, 1972, 646, 166, 889, 3213, 318, 304, 1894, 433, 714, 402, 938, 1267, 3417, 286, 353, 473, 722, 512, 3157, 766, 3267, 275, 2806, 2328, 1732, 457, 1383, 759, 326, 348, 1113, 1243, 814, 2195, 402, 1752, 1615, 152, 1194, 1151, 355, 1737, 445, 53, 53, 2850, 3213, 318, 304, 405, 168, 2318, 328, 281, 1127, 732, 290, 658, 847, 434, 32, 34, 895, 292, 614, 1057, 803, 931, 1999, 996, 652, 726, 2307, 262, 332, 409, 1354, 184, 884, 1000, 555, 459, 687, 361, 273, 275, 1367, 417, 958, 1942, 2772, 667, 433, 174, 3429, 790, 360, 1178, 714, 413, 873, 797, 2087, 1999, 172, 3430, 928, 419, 967, 774, 785, 2378, 1045, 1479, 174, 277, 1732, 457, 1383, 552, 1925, 2806, 548, 789, 413, 2233, 3089, 156, 337, 540, 2630, 750, 338, 541, 836, 1410, 3430, 2895, 412, 919, 625, 1597, 295, 2685, 376, 2378, 1399, 1410, 3430, 419, 2645, 2195, 413, 2665, 797, 334, 2186, 376, 2986, 320, 3430, 761, 166, 596, 317, 1155, 324, 1073, 2688, 319, 2378, 156, 271, 3315, 1570, 1391, 434, 1504, 508, 479, 402, 439, 1608, 717, 261, 45, 2362, 1370, 412, 615, 716, 377, 401, 430, 1222, 320, 533, 281, 2298, 511, 310, 3430, 355, 2123, 1400, 338, 261, 41, 329, 585, 717, 851, 680, 789, 268, 2233, 1752, 3294, 1220, 262, 814, 336, 346, 46, 359, 157]
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