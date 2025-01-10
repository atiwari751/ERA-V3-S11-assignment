import pickle
import regex as re
from tqdm import tqdm

# Read text from a file
with open('text_file_eng.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Define the GPT-2 regex pattern
gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# Apply the regex pattern to the raw text to tokenize it
tokens = re.findall(gpt2pat, text)

# Convert tokens to byte sequences
byte_tokens = [token.encode('utf-8') for token in tokens]

# Create a list of byte sequences, each representing a token
tokens = [list(token) for token in byte_tokens]

def get_stats(token_list):
    """Count frequency of pairs across all tokens"""
    counts = {}
    # Count pairs within each token
    for token in token_list:
        if len(token) < 2:
            continue
        for pair in zip(token, token[1:]):
            counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(token_list, pair, idx):
    """Merge all occurrences of pair within each token"""
    newids = []
    for token in token_list:
        if len(token) < 2:
            newids.append(token)
            continue
            
        new_token = []
        i = 0
        while i < len(token):
            if i < len(token) - 1 and (token[i], token[i+1]) == pair:
                new_token.append(idx)
                i += 2
            else:
                new_token.append(token[i])
                i += 1
        newids.append(new_token)
    return newids

def perform_bpe():
    vocab_size = 1500  # the desired final vocabulary size
    num_merges = vocab_size - 256
    token_list = list(tokens)  # copy so we don't destroy the original list
    
    # Calculate total bytes before compression
    total_bytes_before = sum(len(token) for token in token_list)
    
    merges = {}  # (int, int) -> int
    for i in tqdm(range(num_merges), desc="Performing BPE", unit="merge"):
        stats = get_stats(token_list)
        if not stats:  # No more pairs to merge
            break
            
        # Find most frequent pair
        pair = max(stats, key=stats.get)
        idx = 256 + i
        
        # Perform the merge
        token_list = merge(token_list, pair, idx)
        merges[pair] = idx
    
    # Calculate total bytes after compression
    total_bytes_after = sum(len(token) for token in token_list)
    
    print("---")
    print("Total bytes before:", total_bytes_before)
    print("Total bytes after:", total_bytes_after)
    print(f"Compression ratio: {total_bytes_before / total_bytes_after:.2f}X")
    
    # Flatten for storage, but maintain token boundaries
    flat_ids = []
    for token in token_list:
        flat_ids.extend(token)
    
    return merges, flat_ids, num_merges

if __name__ == "__main__":
    print('---')
    print("length of text:", len(text))
    print('---')
    print("length of tokens:", len(tokens))
    
    # Run BPE and save results
    merges, ids, num_merges = perform_bpe()

    # Save merges and vocab to a file
    with open('bpe_results.pkl', 'wb') as f:
        pickle.dump((merges, ids, num_merges), f)

