import torch as t
import einops
import fancy_einsum as einsum

def conv1d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    '''Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''

    assert x.shape[1] == weights.shape[1], "in_channels must be the same"

    batch, in_channels, width = x.shape
    out_channels, in_channels, kernel_width = weights.shape

    output_width = width - kernel_width + 1

    # first consider batch = 1, out_channels = 1
    x_B, x_IC, x_W = x.stride()
    x_strided_shape = (batch, in_channels, output_width, kernel_width)
    x_new_stride = (x_B, x_IC, x_W, x_W)
    x_strided = t.as_strided(x,x_strided_shape, x_new_stride)


    output = einsum.einsum('b i w k, o i k -> b o w', x_strided, weights)

    # concatenate all resultants --> batch, out_channels, output_width
    return output

def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    '''
    
    X, Y, Z = x.shape

    tens = x.new_full((X, Y, left + Z + right), pad_value)
    tens[..., left:left + Z] = x
    return tens


def conv2d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    '''Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    batch, in_channels, height, width = x.shape
    out_channels, in_channels, kernel_height, kernel_width = weights.shape

    output_height = height - kernel_height + 1
    output_width = width - kernel_width + 1

    xB, xIC, xH, xW = x.stride()

    size = (batch, in_channels, output_height, output_width, kernel_height, kernel_width)
    strides = (xB, xIC, xH, xW, xH, xW)
    x_strided = t.as_strided(x, size, strides)

    # einsum doing black magic here...
    result = einsum.einsum('batch in_channels output_height output_width kernel_height kernel_width, out_channels in_channels kernel_height kernel_width -> batch out_channels output_height output_width', x_strided, weights)
    return result


def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    '''
    B, I, H, W = x.shape

    tens = x.new_full((B, I, top + H + bottom, left + W + right), pad_value)
    tens[...,top:top+H, left:left + W] = x
    return tens