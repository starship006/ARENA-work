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
import utils
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

import utilsd2
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
    



# epochs = 1
# loss_fn = nn.CrossEntropyLoss()
# batch_size = 128

# MODEL_FILENAME = "./w1d2_convnet_mnist.pt"
# device = "cuda" if t.cuda.is_available() else "cpu"

# loss_list, accuracy_list = train_convnet(trainloader, testloader, epochs, loss_fn)



# fig = px.line(y=loss_list, template="simple_white")
# fig.update_layout(title="Cross entropy loss on MNIST", yaxis_range=[0, max(loss_list)])
# fig.show()


# utils.plot_loss_and_accuracy(loss_list, accuracy_list)

MAIN