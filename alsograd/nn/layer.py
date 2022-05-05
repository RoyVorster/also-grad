from typing import Tuple, Sequence, Callable
import numpy as np

from alsograd.core import Parameter
from alsograd.utils import Axis
from alsograd.nn.module import Module


class Linear(Module):
    def __init__(self, in_neurons: int, out_neurons: int):
        super().__init__()

        self.w = Parameter.init(in_neurons, out_neurons)
        self.b = Parameter.zeros(out_neurons)

    def forward(self, x: Parameter) -> Parameter:
        b_shape = [1]*(x.ndim - 1) + [-1]  # Batch reshape
        return x@self.w + self.b.reshape(*b_shape)


class Conv2D(Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 padding: int = 0, kernel_size: int = 3):
        super().__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding

        self.w = Parameter.init(in_channels, out_channels, kernel_size, kernel_size)
        self.b = Parameter.zeros(out_channels)

    def output_size(self, N: int) -> int:
        return 1 + (N + 2*self.padding - self.kernel_size)//self.stride

    def pad_2d(self, x: Parameter) -> Parameter:
        pad = [(0, 0)]*(x.ndim - 2) + [(self.padding, self.padding)]*2
        return x.pad_constant(tuple(pad), value=0)

    def forward(self, x: Parameter) -> Parameter:
        N, _, H, W = x.shape
        H_out, W_out = self.output_size(H), self.output_size(W)

        # Pad
        x_pad = self.pad_2d(x)

        # First filter index
        i_base = np.tile(np.repeat(np.arange(self.kernel_size), self.kernel_size), self.in_channels).reshape((-1, 1))
        i_stride = self.stride*np.repeat(np.arange(H_out), W_out).reshape((1, -1))

        i = (i_base + i_stride)

        # Second filter index
        j_base = np.hstack([np.arange(self.kernel_size)]*self.kernel_size*self.in_channels).reshape((-1, 1))
        j_stride = self.stride*np.hstack([np.arange(W_out)]*H_out).reshape((1, -1))

        j = (j_base + j_stride)

        k = np.repeat(np.arange(self.in_channels), self.kernel_size**2).reshape((-1, 1)).astype(int)

        window = x_pad[:, k, i, j].transpose(order=(1, 2, 0)).reshape(self.in_channels*self.kernel_size**2, -1)

        y = self.w.reshape(self.out_channels, -1)@window + self.b.reshape(-1, 1)
        y = y.reshape(self.out_channels, H_out, W_out, N).transpose(order=(3, 0, 1, 2))

        return y


# Pooling layers
class OpPool2D(Module):
    def __init__(self, f_pool: Callable[[Parameter, Axis], Parameter],
                 kernel_size: Tuple[int, int]):
        super().__init__()

        self.kx, self.ky = kernel_size
        self.f_pool = f_pool  # Hook after pooling

    def forward(self, x: Parameter) -> Parameter:
        H, W = x.shape[-2:]
        xp = x[..., :H - (H % self.ky), :W - (W % self.kx)]

        s, (H, W) = xp.shape[:-2], xp.shape[-2:]
        xp = xp.reshape(*s, H//self.ky, self.ky, H//self.kx, self.kx)

        return self.f_pool(xp, (-3, -1))


def MaxPool2D(kernel_size: Tuple[int, int] = (2, 2)):
    f_pool = lambda x, axis: x.max(axis=axis)
    return OpPool2D(f_pool, kernel_size)


def AvgPool2D(kernel_size: Tuple[int, int] = (2, 2)):
    f_pool = lambda x, axis: x.mean(axis=axis)
    return OpPool2D(f_pool, kernel_size)


# Other
class Sequential(Module):
    def __init__(self, layers: Sequence[Module]):
        super().__init__()

        self.layers = layers

    def forward(self, x: Parameter) -> Parameter:
        for l in self.layers:
            x = l(x)

        return x
