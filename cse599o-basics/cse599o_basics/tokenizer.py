from __future__ import annotations
from typing import Dict, List, Optional, Iterator
import tiktoken

class BPETokenizer:
    """
    A Byte-Pair Encoding (BPE) tokenizer implementation using tiktoken.
    """
    def __init__(
        self, 
        vocab: Dict[int, bytes], 
        merges: List[tuple[bytes, bytes]], 
        special_tokens: Optional[List[str]] = None
    ):
        """
        Initialize the BPE tokenizer with vocabulary, merges, and optional special tokens.
        
        Args:
            vocab: Dictionary mapping token IDs to byte sequences
            merges: List of merge rules as tuples of byte sequences
            special_tokens: Optional list of special tokens to be treated as single tokens
        """
        # Use tiktoken's GPT-2 encoding with special tokens
        self.special_tokens = special_tokens or []
        
        # Create special tokens mapping for tiktoken
        if self.special_tokens:
            # Sort special tokens by length (longest first) to handle overlapping tokens correctly
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_tokens_dict = {token: 50256 + i for i, token in enumerate(sorted_special_tokens)}
            self.tokenizer = tiktoken.Encoding(
                name="gpt2_with_special",
                pat_str=tiktoken.get_encoding("gpt2")._pat_str,
                mergeable_ranks=tiktoken.get_encoding("gpt2")._mergeable_ranks,
                special_tokens=special_tokens_dict
            )
        else:
            self.tokenizer = tiktoken.get_encoding("gpt2")
        
        # Store the vocabulary and merges for reference
        self.vocab = {i: self.tokenizer.decode_single_token_bytes(i) for i in range(self.tokenizer.n_vocab)}
        self.merges = list(self.tokenizer._mergeable_ranks.items())

    def encode(self, text: str) -> List[int]:
        """
        Encode a string into a list of token IDs.
        
        Args:
            text: The input string to tokenize
            
        Returns:
            A list of token IDs
        """
        if not text:
            return []
        
        return self.tokenizer.encode(text, allowed_special="all")

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token IDs back to a string.
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            The decoded string
        """
        if not token_ids:
            return ""
        
        return self.tokenizer.decode(token_ids)

    def encode_iterable(
        self,
        iterable: Iterable[str],
        *,
        output_format: str = "flat",
    ) -> Iterator[int] | Iterator[list[int]]:
        """
        Encode an iterable of strings.
        
        Args:
            iterable: An iterable that yields strings
            output_format: Either "flat" to yield individual token IDs, or "grouped" to yield lists of token IDs
            
        Yields:
            Token IDs (individual ints if flat=True, lists of ints if flat=False)
        """
        flat = output_format == "flat"
        for line in iterable:
            ids = self.encode(line)
            if flat:
                yield from ids
            else:
                yield ids
                
    @classmethod
    def from_serialized(
        cls,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str],
    ):
        """
        Create a BPETokenizer instance from serialized data.
        
        Args:
            vocab: Dictionary mapping token IDs to byte sequences
            merges: List of merge rules as tuples of byte sequences
            special_tokens: List of special tokens to be treated as single tokens
            
        Returns:
            A new BPETokenizer instance
        """
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
