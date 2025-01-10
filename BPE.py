import pickle
from tqdm import tqdm  # Import tqdm for progress bar

# Read text from a file
with open('text_file.txt', 'r', encoding='utf-8') as file:
    text = file.read()

tokens = text.encode("utf-8")  # raw bytes
tokens = list(map(int, tokens))  # convert to a list of integers in range 0..255 for convenience

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    # in the list of ints (ids), replace all consecutive occurrences of pair with the new token idx
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
    vocab_size = 3500  # the desired final vocabulary size
    num_merges = vocab_size - 256
    ids = list(tokens)  # copy so we don't destroy the original list

    merges = {}  # (int, int) -> int
    # Use tqdm to add a progress bar
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

