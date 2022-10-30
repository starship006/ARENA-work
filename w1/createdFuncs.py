# a place to store previously written functions for
# easy accessibility throughout the days
import torch as t
import numpy as np
from torch import nn

def single_head_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor) -> t.Tensor:
    '''
    Should return the results of self-attention (see the "Self-Attention in Detail" section of the Illustrated Transformer).

    With this function, you can ignore masking.

    Q: shape (seq_len, head_size)
    K: shape (seq_len, head_size)
    V: shape (seq_len, value_size)

    Return: shape (seq_len, value_size)
    '''

    # second step - calculate a "score"
    score = Q@(K.T) # shape (seq_len,seq_len)
    # third step - divide score by dimensionality
    score = score / np.sqrt(Q.shape[-1])
    # fourth step - softmax
    score = nn.functional.softmax(score, dim=-1)

    # fifth step - multiply each value vector by the softmax score
    z = score@V
    return z


def single_head_masked_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor) -> t.Tensor:
    '''
    Should return the results of masked self-attention.

    See "The Decoder Side" section of the Illustrated Transformer for an explanation of masking.

    Q: shape (seq_len, head_size)
    K: shape (seq_len, head_size)
    V: shape (seq_len, value_size)

    Return: shape (seq_len, value_size)
    '''
    # second step - calculate a "score"
    score = Q@(K.T) # shape (seq_len,seq_len)
    # third step - divide score by dimensionality
    score = score / np.sqrt(Q.shape[-1])
    # MASKING IN BETWEEN!
    mask = t.ones(score.shape)
    for i, x in enumerate(mask):
        x[i+1:] = -t.inf
    score = score * mask
    # fourth step - softmax
    score = nn.functional.softmax(score, dim=-1)

    # fifth step - multiply each value vector by the softmax score
    z = score@V
    return z
