# Read text from a file
with open('text_file.txt', 'r', encoding='utf-8') as file:
    text = file.read()

tokens = text.encode("utf-8")  # raw bytes
tokens = list(map(int, tokens))  # convert to a list of integers in range 0..255 for convenience

print('---')
print("length of text:", len(text))
print('---')
#print(tokens)
print('---')
print("length of tokens:", len(tokens))

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]): # Pythonic way to iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

stats = get_stats(tokens)
print('---')
# print(stats)
#print(sorted(((v,k) for k,v in stats.items()), reverse=True))

print('---')
top_pair = max(stats, key=stats.get)
print(top_pair)

#print(chr(224), chr(164))