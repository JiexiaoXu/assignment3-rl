import tiktoken
import numpy as np
from cse599o_basics.tokenizer import BPETokenizer
import os

def tokenize_data(file_path, output_path):
    encoding = tiktoken.get_encoding("gpt2")
    vocab = {i: encoding.decode_single_token_bytes(i) for i in range(encoding.n_vocab)}
    merges = list(encoding._mergeable_ranks.items())
    special_tokens = ["<|endoftext|>"]

    tokenizer = BPETokenizer(vocab, merges, special_tokens)
    
    dtype = np.uint16
    chunk_size = 50 * 1024 * 1024  # 50MB chunk size
    token_processed = 0
    buffer = []
    
    with open(file_path, "r", encoding="utf-8") as f , open(output_path, "wb") as out_f:
        for token_id in tokenizer.encode_iterable(f, output_format="flat"):
            buffer.append(token_id)
            
            if len(buffer) >= chunk_size:
                arr = np.array(buffer, dtype=dtype)
                out_f.write(arr.tobytes())
                token_processed += len(buffer)
                buffer.clear()
        
        if buffer:
            arr = np.array(buffer, dtype=dtype)
            out_f.write(arr.tobytes())
            token_processed += len(buffer)
            buffer.clear()

        print(f"Total tokens processed: {token_processed}")

def verify(file_path, data_path, dtype=np.uint16):
    arr = np.memmap(data_path, dtype=dtype, mode="r")
    print(f"Total tokens in {data_path}: {len(arr)}")
    print(f"First 10 tokens: {arr[:10]}")
    print(f"Last 10 tokens: {arr[-10:]}")
    
       
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tokenize text data using BPE tokenizer.")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the raw text data file."
    )
    parser.add_argument(
        "--output_path", type=str, default="tokenized_data.bin", help="Path to save the tokenized data."
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify the tokenized data after processing."
    )

    args = parser.parse_args()

    if args.verify:
        verify(args.output_path)
    else:
        tokenize_data(args.data_path, args.output_path)
        if args.verify:
            verify(args.output_path)
        print(f"Tokenized data saved to {args.output_path}")
    