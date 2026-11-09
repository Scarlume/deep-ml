class CharTokenizer:
    def __init__(self, text: str):
        """
        Build a character-level tokenizer from the input text.
        
        Args:
            text: A string used to build the vocabulary.
        """
        # Your code here
        
        self.stoi = {'<BOS>':0, '<EOS>':1}
        self.itos = {0: '<BOS>' ,1: '<EOS>'}
        self.vocab_size = 2
        sorted_chars = sorted(set(text))
        for ch in sorted_chars:
            self.stoi[ch]=self.vocab_size
            self.itos[self.vocab_size]=ch
            self.vocab_size+=1


    def encode(self, text: str) -> list:
        """
        Encode a string into a list of token indices.
        
        Args:
            text: The string to encode.
        Returns:
            List of integer indices.
        """
        # Your code here
        res = [0]
        for ch in text:
            res.append(self.stoi[ch])
        res.append(1)
        return res

    def decode(self, indices: list) -> str:
        """
        Decode a list of token indices back into a string.
        
        Args:
            indices: List of integer indices.
        Returns:
            Decoded string.
        """
        # Your code here
        res = []
        for i in indices:
            res.append(self.itos[i])
        return ''.join(res)
