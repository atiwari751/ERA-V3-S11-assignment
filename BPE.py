import pickle
import regex as re
from tqdm import tqdm

# Read text from a file
with open('text_file_eng.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Define the GPT-2 regex pattern
gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# Tokenize the text using the regex pattern
tokens = re.findall(gpt2pat, text)

# Convert tokens to byte sequences
byte_tokens = [token.encode('utf-8') for token in tokens]

# Flatten the list of byte sequences into a single list of bytes
tokens = [b for token in byte_tokens for b in token]

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def perform_bpe():
    vocab_size = 1500  # the desired final vocabulary size
    num_merges = vocab_size - 256
    ids = list(tokens)  # copy so we don't destroy the original list

    merges = {}  # (int, int) -> int
    for i in tqdm(range(num_merges), desc="Performing BPE", unit="merge"):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        ids = merge(ids, pair, idx)
        merges[pair] = idx

    print("---")
    print("ids length:", len(ids))
    print(f"compression ratio: {len(tokens) / len(ids):.2f}X")
    
    return merges, ids, num_merges

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

