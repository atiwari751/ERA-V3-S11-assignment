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
set_of_ids = [1044, 1283, 262, 260, 314, 266, 284, 259, 261, 262, 263, 308, 263, 267, 279, 263, 410, 660, 275, 318, 319, 318, 544, 263, 319, 338, 269, 286, 266, 261, 265, 284, 259, 533, 262, 265, 320, 260, 271, 265, 261, 277, 282, 916, 260, 270, 263, 507, 322, 278, 336, 318, 262, 260, 286, 259, 278, 273, 260, 1097, 259, 267, 260, 291, 259, 267, 260, 298, 259, 270, 260, 343, 557, 273, 287, 341, 372, 260, 592, 296, 262, 260, 571, 280, 342, 265, 301, 277, 282, 798, 263, 274, 344, 369, 603, 259, 269, 259, 292, 266, 270, 265, 270, 263, 262, 263, 417, 271, 266, 276, 259, 272, 266, 269, 291, 259, 278, 259, 833, 265, 290, 266, 283, 273, 260, 279, 281, 569, 262, 263, 292, 300, 261, 263, 314, 274, 421, 313, 328, 314, 266, 284, 259, 261, 262, 260, 1277, 827, 265, 275, 924, 259, 267, 262, 282, 283, 265, 261, 545, 314, 279, 264, 290, 263, 269, 415, 271, 924, 259, 271, 259, 274, 344, 294, 260, 269, 266, 267, 279, 266, 572, 260, 273, 259, 270, 294, 268, 622, 259, 319, 280, 267, 259, 284, 296, 262, 260, 292, 338, 261, 259, 267, 506, 1283, 262, 268, 462, 259, 437, 259, 262, 266, 380, 367, 260, 314, 266, 284, 259, 261, 262, 260, 340, 376, 294, 268, 289, 296, 277, 282, 273, 260, 343, 274, 489, 291, 266, 526, 259, 286, 259, 278, 924, 259, 271, 259, 273, 300, 305, 263, 273, 260, 328, 651, 273, 260, 262, 259, 287, 292, 266, 275, 259, 361, 259, 262, 266, 340, 265, 276, 296, 267, 260, 367, 259, 592, 766, 294, 266, 275, 259, 274, 312, 10, 10, 53, 53, 483, 265, 320, 263, 275, 506, 1283, 286, 260, 286, 265, 275, 300, 290, 322, 270, 353, 267, 265, 301, 265, 261, 263, 262, 268, 413, 259, 275, 366, 32, 34, 519, 259, 261, 260, 273, 259, 440, 260, 321, 260, 379, 263, 285, 259, 261, 274, 280, 357, 274, 312, 294, 260, 269, 266, 267, 286, 259, 278, 262, 476, 260, 262, 260, 285, 259, 261, 260, 277, 282, 540, 260, 291, 265, 275, 259, 283, 259, 286, 266, 261, 259, 302, 259, 389, 269, 285, 259, 271, 321, 260, 323, 263, 262, 266, 419, 259, 462, 259, 924, 259, 267, 262, 260, 292, 266, 267, 274, 263, 437, 330, 433, 375, 379, 263, 285, 259, 261, 285, 300, 329, 545, 314, 431, 308, 268, 340, 265, 276, 296, 267, 260, 375, 282, 413, 259, 275, 259, 262, 266, 340, 265, 276, 282, 924, 259, 271, 259, 273, 300, 305, 263, 277, 282, 375, 259, 261, 259, 286, 259, 278, 345, 335, 277, 266, 270, 330, 792, 266, 299, 375, 282, 291, 259, 278, 259, 833, 265, 290, 266, 283, 277, 282, 262, 266, 272, 263, 292, 300, 391, 260, 285, 300, 329, 314, 291, 259, 341, 292, 260, 607, 259, 319, 259, 276, 266, 299, 46, 380, 318, 273, 260, 375, 282, 292, 300, 391, 260, 285, 300, 329, 314, 310, 260, 290, 292, 266, 275, 259, 361, 330, 328, 513, 899, 260, 279, 318, 305, 273, 260, 1067, 285, 300, 329, 296, 262, 259, 292, 338, 261, 259, 262, 266, 275, 330, 313, 332, 266, 261, 297, 264, 271, 277, 282, 375, 282, 291, 268, 262, 259, 518, 413, 259, 275, 259, 361, 366, 298, 268, 369, 323, 259, 262, 266, 327, 259, 741, 592, 45, 1133, 291, 259, 341, 534, 265, 284, 260, 269, 265, 515, 262, 260, 292, 338, 261, 259, 267, 285, 263, 901, 399, 394, 295, 300, 329, 294, 260, 643, 626, 266, 391, 41, 262, 268, 375, 592, 314, 345, 335, 277, 266, 270, 260, 792, 266, 299, 340, 265, 276, 296, 267, 260, 375, 259, 261, 260, 286, 259, 278, 262, 259, 287, 292, 266, 299, 46, 1396]
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