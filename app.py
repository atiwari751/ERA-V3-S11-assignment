import streamlit as st
from tokenizer import HindiTokenizer

# Initialize tokenizer
@st.cache_resource
def load_tokenizer():
    return HindiTokenizer()

def format_token_ids(token_ids):
    # Format token IDs in a readable way, 10 per line
    lines = []
    for i in range(0, len(token_ids), 10):
        line = token_ids[i:i+10]
        lines.append(' '.join(str(id) for id in line))
    return '\n'.join(lines)

def format_hindi_tokens(tokens):
    # Join tokens with double spaces
    return '  '.join(tokens)

def main():
    st.title("Hindi Text Tokenizer")
    
    tokenizer = load_tokenizer()
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Word Count")
    with col2:
        st.subheader("Compression Ratio")
    with col3:
        st.subheader("BPE Tokens")  # Renamed to clarify these are post-BPE tokens
    
    # Text input
    st.subheader("Input Text:")
    text_input = st.text_area(
        label="Input Hindi text", 
        height=150, 
        key="input",
        label_visibility="collapsed"
    )
    
    if st.button("Tokenize"):
        if text_input:
            # Get tokens and IDs
            token_ids, original_tokens, decoded_tokens = tokenizer.tokenize(text_input)
            
            # Calculate metrics
            word_count = len(text_input.split())
            original_bytes = sum(len(token.encode('utf-8')) for token in original_tokens)
            compression_ratio = original_bytes / len(token_ids)
            
            # Update metrics
            col1.write(f"{word_count}")
            col2.write(f"{compression_ratio:.2f}X")
            col3.write(f"{len(token_ids)}")  # This is post-BPE token count
            
            # Optional: Display both token counts for comparison
            st.caption(f"Initial tokens (after regex): {len(original_tokens)}")
            st.caption(f"Final tokens (after BPE): {len(token_ids)}")
            
            # Display token IDs in a formatted way
            st.subheader("Token IDs:")
            st.text_area(
                label="Generated token IDs",
                value=format_token_ids(token_ids), 
                height=150, 
                key="ids",
                label_visibility="collapsed"
            )
            
            # Display decoded tokens with tab separation
            st.subheader("Tokenized Text:")
            st.text_area(
                label="Tokenized output",
                value='\t'.join(decoded_tokens), 
                height=150, 
                key="tokens",
                label_visibility="collapsed"
            )
        else:
            st.warning("Please enter some text to tokenize.")

if __name__ == "__main__":
    main() 