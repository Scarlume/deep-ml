import torch
import torch.nn.functional as F
from typing import Tuple

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)

    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)

    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    d_k = Q.shape[-1]
    att_score = Q @ K.transpose(0,1)
    softm = F.softmax(att_score/(d_k**0.5)) # [sen_q, d_k]
    return softm @ V

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads

    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    seq_len, d_model = Q.shape
    d_k = d_model // n_heads
    Q = Q.reshape(seq_len, n_heads, d_k).permute(1,0,2).contiguous() #(n_heads, seq_len, d_k)
    K = K.reshape(seq_len, n_heads, d_k).permute(1,2,0).contiguous() # (n_heads, d_k, seq_len)
    V = V.reshape(seq_len, n_heads, d_k).permute(1,0,2).contiguous() #(n_heads, seq_len, d_k)

    att_score =(Q @ K)/(d_k**0.5) # (n_heads, seq_len, seq_Len)
    softm = att_score.softmax(dim=-1)
    res = softm @ V
    res = res.permute(1,0,2).reshape(seq_len,-1).contiguous()
    return res
    