from typing import List, Optional


class Tokenizer:
    
    def __init__(self, vocab: Optional[List[str]] = None, 
                 corpus: Optional[str] = None) -> None:
        if not (vocab or corpus):
            raise ValueError("Either vocab or corpus must be provided")

        if not vocab:
            vocab = list(set(corpus))
        self.corpus = ''.join(vocab)
        self.vocab = sorted(vocab)
        self.vocab_size = len(self.vocab)
        
        self.s2i = {char: i for i, char in enumerate(self.vocab)}
        self.i2s = {i: char for i, char in enumerate(self.vocab)}
    
    def encode(self, text: str) -> List[int]:
        return [self.s2i[char] for char in text]

    def decode(self, tokens: List[int]) -> str:
        return ''.join([self.i2s[i] for i in tokens])
            
