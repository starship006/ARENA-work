from tkinter.tix import MAIN
from typing import Union, Optional, Callable
import torch as t
import PIL 
from PIL import Image 
import json 
from pathlib import Path 
from typing import Union, Tuple, Callable, Optional 
import plotly.graph_objects as go 
import plotly.express as px 
from plotly.subplots import make_subplots 
IntOrPair = Union[int, tuple[int, int]]
Pair = tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
    '''Convert v to a pair of int, if it isn't already.'''
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)

def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    '''
    B, I, H, W = x.shape

    tens = x.new_full((B, I, top + H + bottom, left + W + right), pad_value)
    tens[...,top:top+H, left:left + W] = x
    return tens

import torch.nn as nn

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 1):
        super().__init__()
        if stride == None:
            stride = kernel_size
        kernel_size = force_pair(kernel_size)
        stride = force_pair(stride)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = force_pair(padding)

        
    def forward(self, x: t.Tensor) -> t.Tensor:
        new_x = pad2d(x, self.padding[1], self.padding[1], self.padding[0], self.padding[0], -99999999999)
        batch, in_channels, height, width = x.shape
        kernel_height = self.kernel_size[0]
        kernel_width = self.kernel_size[1]

        xB, xIB, xH, xW = new_x.stride()

        output_width = 1 + (width + 2 * self.padding[1] - kernel_width) // self.stride[1]
        output_height = 1 + (height + 2 * self.padding[0] - kernel_height) // self.stride[0]
        size = (batch, in_channels, output_height, output_width, kernel_height, kernel_width)
        strides = (xB, xIB, xH * self.stride[0], xW * self.stride[1],  xH, xW)
        new_x = t.as_strided(new_x, size, strides)

        new_x = t.amax(new_x, 5)
        new_x = t.amax(new_x, 4)
        return new_x



    def extra_repr(self) -> str:
        '''Add additional information to the string representation of this class.'''
        output = ""
        output += "kernel_size = " + str(self.kernel_size)
        output += " stride = " + str(self.stride)
        output += " padding = " + str(self.padding)
        return output
    

class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.maximum(x, t.tensor(0.0))


class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        '''Flatten out dimensions from start_dim to end_dim, inclusive of both.
        '''
        # get start and end dimensions
        start, end = self.start_dim, self.end_dim
        if (start < 0):
            start = len(input.shape) + start
        if (end < 0):
            end = len(input.shape) + end

        # get shape of output
        shape = []
        for index in range(len(input.shape)):
            shape.append(input.shape[index])
        
        # on start_dim element, multiply each value up until end_dim, removing them each time
        to_remove = end - start
        for i in range(to_remove):
            shape[start] *= shape.pop(start + 1)
        shape = tuple(shape)

        reshaped = t.reshape(input,shape)
        return reshaped

    def extra_repr(self) -> str:
        output = ""
        output += "start_dim = " + str(self.start_dim) + " "
        output += "end_dim = " + str(self.end_dim)


        return output


from audioop import bias
import math
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        '''A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        '''
        super().__init__()
        bound = -1 / math.sqrt(in_features)
        self.weight = nn.Parameter(t.FloatTensor(out_features,in_features).uniform_(bound, -bound))
        if bias:
            self.bias = nn.Parameter(t.FloatTensor(out_features).uniform_(bound, -bound))
        else:
            self.bias = None

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        result = x@self.weight.T
        if self.bias != None:
            result = result + self.bias
        return result

    def extra_repr(self) -> str:
        output = ""
        output += "weight.shape = " + str(self.weight.shape) + " "
        if self.bias != None:
            output += "bias.shape = " + str(self.bias.shape)
        else:
            output += "bias = None"
        return output


from fancy_einsum import einsum
class Conv2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0
    ):
        '''
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        '''
        super().__init__()
        kernel_size = force_pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        bound = -1 / math.sqrt(kernel_size[0] * kernel_size[1] * in_channels)
        self.weight = nn.Parameter(t.FloatTensor(out_channels, in_channels, kernel_size[0], kernel_size[1]).uniform_(bound, -bound))
        self.stride = force_pair(stride)
        self.padding = force_pair(padding)

    def forward(self, x: t.Tensor) -> t.Tensor:
        # setup a strided tensor to represent input
        new_x = pad2d(x, self.padding[1], self.padding[1], self.padding[0], self.padding[0], 0)
        batch, in_channels, height, width = x.shape
        out_channels, in_channels, kernel_height, kernel_width = self.weight.shape
        xB, xIB, xH, xW = new_x.stride()
        output_width = 1 + (width + 2 * self.padding[1] - kernel_width) // self.stride[1]
        output_height = 1 + (height + 2 * self.padding[0] - kernel_height) // self.stride[0]
        size = (batch, in_channels, output_height, output_width, kernel_height, kernel_width)
        strides = (xB, xIB, xH * self.stride[0], xW * self.stride[1],  xH, xW)
        new_x = t.as_strided(new_x, size, strides)

        return einsum('batch in_channels output_height output_width kernel_height kernel_width, out_channels in_channels kernel_height kernel_width -> batch out_channels output_height output_width ', new_x, self.weight)

    
    def extra_repr(self) -> str:
        output = ""
        output += "weights shape = " + str(self.weight.shape) + " "
        output += "stride = " + str(self.stride) + " "
        output += "padding = " + str(self.padding) + " "
        return output

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 32, 3, 1, 1)
        self.ReLU1 = ReLU()
        self.max1 = MaxPool2d(2,2,0)
        self.conv2 = Conv2d(32, 64, 3, 1, 1)
        self.ReLU2 = ReLU()
        self.max2 = MaxPool2d(2,2,0)
        self.flatten = Flatten()
        self.lin1 = Linear(3136, 128)
        self.lin2 = Linear(128, 10)

    def forward(self, x: t.Tensor) -> t.Tensor:
        funcsToApply = [self.conv1, self.ReLU1, self.max1, self.conv2, self.ReLU2, self.max2, self.flatten, self.lin1, self.lin2]
        result = x
        for f in funcsToApply:
            print(f)
            result = f(x)
        return result


    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.max1(self.ReLU1(self.conv1(x)))
        x = self.max2(self.ReLU2(self.conv2(x)))
        x = self.lin2(self.lin1(self.flatten(x)))
        return x


import torchvision 
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader 
import tqdm
from tqdm.notebook import tqdm_notebook

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
# set up training data

testset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
testloader = DataLoader(testset, batch_size=128, shuffle=True)

device = t.device

def train_convnet(trainloader: DataLoader, testloader: DataLoader, epochs: int, loss_fn: Callable) -> list:
    '''
    Defines a ConvNet using our previous code, and trains it on the data in trainloader.

    Returns tuple of (loss_list, accuracy_list), where accuracy_list contains the fraction of accurate classifications on the test set, at the end of each epoch.
    '''
    model = ConvNet().to(device).train()
    optimizer = t.optim.Adam(model.parameters())
    loss_list = []
    accuracy_list = []

    for epoch in tqdm_notebook(range(epochs)):

        
        for (x, y) in tqdm_notebook(trainloader, leave=False):

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

            # get accuracy
            totalCorrect = 0.0
            count = 128
            X_batch, y_batch = next(iter(testloader))
            y_hat = t.argmax(model(X_batch),1)
            totalCorrect = (y_hat == y_batch).float().sum()
            accuracy = totalCorrect / count

            print("totalCorrect =  " + str(totalCorrect) + ", count = " + str(count) )
            print(f"Epoch {epoch}/{epochs}, accuracy is {accuracy:.6f}")
            accuracy_list.append(accuracy)

        print(f"Epoch {epoch}/{epochs}, train loss is {loss:.6f}")




    print(f"Saving model to: {MODEL_FILENAME}")
    t.save(model, MODEL_FILENAME)
    return (loss_list, accuracy_list)
    

######################################################
###########################

class Sequential(nn.Module):
    def __init__(self, *modules: nn.Module):
        super().__init__()
        for i, mod in enumerate(modules):
            self.add_module(str(i), mod)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Chain each module together, with the output from one feeding into the next one.'''
        for mod in self._modules.values():
            x = mod(x)
        return x

model = Sequential(Conv2d(3,10,3,1,0),
ReLU(),
nn.Linear(10,5))

print(model)

from math import gamma
from operator import truediv
import einops
from tkinter.tix import MAIN


class BatchNorm2d(nn.Module):
    running_mean: t.Tensor         # shape: (num_features,)
    running_var: t.Tensor          # shape: (num_features,)
    num_batches_tracked: t.Tensor  # shape: ()

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        '''Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        '''
        super().__init__()
        #print("num features is " + str(num_features))
        running_mean = t.zeros(num_features)
        running_var = t.ones(num_features)
        num_batches_tracked = t.tensor(0)
        
        self.weight = nn.Parameter(t.ones(num_features)) # tracks gamma
        self.bias = nn.Parameter(t.zeros(num_features)) # tracks beta
        self.register_buffer('running_mean', running_mean) # holds mean across channels
        self.register_buffer('running_var', running_var) # holds variance across channels
        self.register_buffer('num_batches_tracked', num_batches_tracked) # stores num batches tracked
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps


    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''
        if self.training:
            # update running mean
            mean = t.mean(x, (0,2,3), keepdim = False)
            self.running_mean = (self.momentum) * mean + (1 - self.momentum) * self.running_mean
            # update running variance
            var = t.var(x, (0,2,3), keepdim = False, unbiased=False)
            self.running_var = (self.momentum) * var + (1 - self.momentum) * self.running_var
            self.num_batches_tracked += 1
        else:
            mean = self.running_mean
            var = self.running_var

        mean = einops.rearrange(mean, 'c -> 1 c 1 1')
        var = einops.rearrange(var, 'c -> 1 c 1 1')
        gamma = einops.rearrange(self.weight, 'c -> 1 c 1 1')
        beta = einops.rearrange(self.bias, 'c -> 1 c 1 1')

        returnX = ((x - mean) / (t.sqrt(var + self.eps))) * gamma + beta
        return returnX

       
    # credit to callum for this
    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["num_features", "eps", "momentum"]])

class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        '''

        return t.mean(x,(2,3))


class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        '''A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        '''

        super().__init__()
        
        left_branch = nn.Sequential(
            Conv2d(in_feats, out_feats, 3, first_stride, 1), # TODO: why is padding 1? not zero?
            BatchNorm2d(out_feats),
            ReLU(),
            Conv2d(out_feats, out_feats, 3, 1, 1),
            BatchNorm2d(out_feats)
        )

        if first_stride > 1:
            right_branch = nn.Sequential(
                Conv2d(in_feats, out_feats, 1, first_stride, 0),
                BatchNorm2d(out_feats)
            )
        else:
            right_branch = nn.Sequential(
                nn.Identity()
            )

        self.left = left_branch
        self.right = right_branch
        self.relu = ReLU()
        

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        leftOutput = self.left.forward(x)
        rightOutput = self.right.forward(x)
        return self.relu.forward(leftOutput + rightOutput)

class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        '''An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride.'''
        super().__init__()
        self.firstRes = ResidualBlock(in_feats,out_feats,first_stride)
        self.otherRes = nn.ModuleList([ResidualBlock(out_feats, out_feats,1) for i in range(n_blocks - 1)])
        # TODO: convert back to normal list if this doesn't work!
        

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Compute the forward pass.
        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        result = self.firstRes.forward(x)
        for l in self.otherRes:
            result = l(result)
        
        return result

import pwd
class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        BlockGroups = []
        for i, j in enumerate(n_blocks_per_group):
            n_blocks = j
            if i > 1:
                in_feats = out_features_per_group[i-1]
            else:
                in_feats = 64
            out_feats = out_features_per_group[i]
            first_stride = first_strides_per_group[i]
            BlockGroups.append(BlockGroup(n_blocks, in_feats, out_feats, first_stride))


        self.preBlockGroups = nn.Sequential(
            Conv2d(3,64,7,2,3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(3,2),
        )
        self.layers = nn.Sequential(
            *BlockGroups
        )

        self.postBlockGroups = nn.Sequential(
            AveragePool(),
            Flatten(),
            Linear(out_features_per_group[-1], 1000)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)

        Return: shape (batch, n_classes)
        '''
        #print("HERE")
        y = self.preBlockGroups(x)
        #print("went through pre block groups")
        y = self.layers(y)
        #print("went through post block groups")
        y = self.postBlockGroups(y)
        return y
