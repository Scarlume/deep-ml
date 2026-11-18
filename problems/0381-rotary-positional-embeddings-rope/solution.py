import torch


def rotate_half(x:torch.Tensor):
    """
    x = [x0,x1,x2,x3]
    ->
    x = [-x1,x0,-x3,x2]
    """

    x_even = x[...,::2]
    x_odd = x[...,1::2] # (seq_len,d/2)
    x = torch.stack((-x_odd,x_even),dim=-1) # (seq_len,d/2,2)
    return x.flatten(-2)



def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """
    Apply Rotary Positional Embeddings (RoPE) to input embeddings.
    
    Args:
        x: Input embeddings of shape (seq_len, d), d must be even
        positions: Position indices of shape (seq_len,)
        base: Base for frequency computation (default: 10000.0)
    
    Returns:
        Embeddings with rotary positional encoding applied, shape (seq_len, d)
    """
    # Your code here
    seq_len, dim = x.shape
    # (dim/2,)
    freqs = base ** (-torch.arange(0, dim, 2)/dim)
    # angles: (seq_len, dim/2)
    angles = torch.outer(positions,freqs) 
    # angles: (seq_len, dim)
    angles = torch.repeat_interleave(angles,2,dim=-1)
    return torch.cos(angles)*x+torch.sin(angles)*rotate_half(x)


