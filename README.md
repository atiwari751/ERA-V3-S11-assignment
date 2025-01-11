# Hindi BPE Tokenizer

A Byte-Pair Encoding tokenizer for Hindi text, implemented using Streamlit.

Hugging face app - https://huggingface.co/spaces/atiwari751/Hindi-tokenizer

## Features
- Tokenizes Hindi text using BPE algorithm
- Visualizes the tokenization process
- Supports custom vocabulary

## Dataset

There were 2 prime sources which were combined in the .txt file - 

- https://www.kaggle.com/datasets/disisbig/hindi-text-short-summarization-corpus (test dataset)
- https://hindi.newslaundry.com/report 

The raw text in the dataset was split into fragments using regex, and these fragments were sent to the BPE algorithm. Some stats - 

Total words in the dataset:      1150937

Total subwords after regex:      1354962

## Vocabulary and Compression Ratio

Total vocabulary: 4000 tokens

Total bytes before: 14659421

Total bytes after: 1889786

Compression ratio: 7.76X
