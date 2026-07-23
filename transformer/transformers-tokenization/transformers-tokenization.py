import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {"<PAD>":0, "<UNK>":1, "<BOS>":2, "<EOS>":3}
        self.id_to_word: Dict[int, str] = {0:"<PAD>", 1:"<UNK>", 2:"<BOS>", 3:"<EOS>"}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        unique_words = set()
        for text in texts:
            words = text.lower().split()
            unique_words.update(words)
        
        id = 4       
        for word in sorted(unique_words):
            self.word_to_id[word] = id
            self.id_to_word[id] = word
            id += 1

        self.vocab_size = id
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        words = text.lower().split()
        unk_id = self.word_to_id[self.unk_token]
        encoded_text = [self.word_to_id.get(word, unk_id) for word in words]
        return encoded_text
                
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        decoded_text = [self.id_to_word.get(idx, self.unk_token) for idx in ids]
        return " ".join(decoded_text)
