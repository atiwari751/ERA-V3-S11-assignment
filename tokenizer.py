import pickle
import regex as re
from typing import List, Tuple

class HindiTokenizer:
    def __init__(self, model_path: str = 'bpe_results.pkl'):
        # Load the BPE model
        with open(model_path, 'rb') as f:
            self.merges, self.ids, self.num_merges = pickle.load(f)
            
        # Initialize vocabulary
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
            
        # Hindi-focused pattern
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{N}+| ?(?:[\u0904-\u0939\u093d-\u093d\u0950-\u0950\u0958-\u0961\u0970-\u097f\ua8f2-\ua8fe\U00011b00-\U00011b09\u1cd3-\u1cd3\u1ce9-\u1cec\u1cee-\u1cf3\u1cf5-\u1cf6\u1cfa-\u1cfa][\u0900-\u0903\u093a-\u093c\u093e-\u094f\u0951-\u0957\u0962-\u0963\ua8e0-\ua8f1\ua8ff-\ua8ff\u1cd0-\u1cd2\u1cd4-\u1ce8\u1ced-\u1ced\u1cf4-\u1cf4\u1cf7-\u1cf9]*)+| ?\p{L}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    def tokenize(self, text: str) -> Tuple[List[int], List[str], List[str]]:
        # Get initial tokens using regex
        tokens = re.findall(self.pattern, text)
        
        # Convert tokens to byte sequences and maintain grouping
        byte_tokens = [token.encode('utf-8') for token in tokens]
        token_list = [list(token) for token in byte_tokens]
        
        # Process each token
        final_tokens = []
        for token in token_list:
            current_token = list(token)
            while len(current_token) >= 2:
                stats = self._get_stats([current_token])
                if not stats:
                    break
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                idx = self.merges[pair]
                current_token = self._merge([current_token], pair, idx)[0]
            final_tokens.extend(current_token)
        
        # Decode the tokens
        decoded_tokens = [self.vocab[idx].decode("utf-8", errors="replace") for idx in final_tokens]
        
        return final_tokens, tokens, decoded_tokens
    
    def _get_stats(self, token_list):
        """Count frequency of pairs across all tokens"""
        counts = {}
        for token in token_list:
            if len(token) < 2:
                continue
            for pair in zip(token, token[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def _merge(self, token_list, pair, idx):
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