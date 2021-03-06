from __future__ import annotations

from typing import Tuple, Sequence, Callable, List
from enum import Enum
import numpy as np

from alsograd.core import Parameter
from alsograd.utils import Axis
from alsograd.nn.module import Module
import alsograd.nn.functions as F


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias: bool = True):
        super().__init__()

        self.bias = bias

        self.w = Parameter.init(in_size, out_size)
        if self.bias:
            self.b = Parameter.zeros(out_size)

    def forward(self, x: Parameter) -> Parameter:
        if self.bias:
            return F.addmm(x, self.b, self.w)

        return x@self.w

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
class Pool2D(Module):
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
    return Pool2D(f_pool, kernel_size)


def AvgPool2D(kernel_size: Tuple[int, int] = (2, 2)):
    f_pool = lambda x, axis: x.mean(axis=axis)
    return Pool2D(f_pool, kernel_size)


class RNNModule(Module):
    pass


class RNN(RNNModule):
    def __init__(self, in_size: int, hidden_size: int):
        super().__init__()

        self.hidden_size = hidden_size

        std = 1/np.sqrt(hidden_size)
        self.w_xh = Parameter.uniform(in_size, hidden_size, lo=-std, hi=std)
        self.b_xh = Parameter.uniform(hidden_size, lo=-std, hi=std)

        self.w_hh = Parameter.uniform(hidden_size, hidden_size, lo=-std, hi=std)
        self.b_hh = Parameter.uniform(hidden_size, lo=-std, hi=std)

    def forward(self, x: Parameter) -> Parameter:
        x = x.transpose(order=(1, 0, 2))
        T, N, _ = x.shape

        h = Parameter.zeros(N, self.hidden_size)

        ys: List[Parameter] = []
        for t in range(T):
            xh = F.addmm(x[t, ...], self.b_xh, self.w_xh)
            hh = F.addmm(h, self.b_hh, self.w_hh)

            h = (xh + hh).tanh()
            ys.append(h)

        return F.stack(ys, axis=0).transpose(order=(1, 0, 2))


class GRU(RNNModule):
    def __init__(self, in_size: int, hidden_size: int):
        super().__init__()

        self.hidden_size = hidden_size
        self.p_size = 3*hidden_size

        std = 1/np.sqrt(hidden_size)
        self.w_xh = Parameter.uniform(in_size, self.p_size, lo=-std, hi=std)
        self.b_xh = Parameter.uniform(self.p_size, lo=-std, hi=std)

        self.w_hh = Parameter.uniform(hidden_size, self.p_size, lo=-std, hi=std)
        self.b_hh = Parameter.uniform(self.p_size, lo=-std, hi=std)

    def forward(self, x: Parameter) -> Parameter:
        x = x.transpose(order=(1, 0, 2))
        T, N, _ = x.shape

        H = self.hidden_size
        h = Parameter.zeros(N, H, requires_grad=False)

        ys: List[Parameter] = []
        for t in range(T):
            xh = F.addmm(x[t, ...], self.b_xh, self.w_xh)
            hh = F.addmm(h, self.b_hh, self.w_hh)

            rt = F.sigmoid(xh[:, :H] + hh[:, :H])
            zt = F.sigmoid(xh[:, H:2*H] + hh[:, H:2*H])
            nt = (xh[:, 2*H:] + rt*hh[:, 2*H:]).tanh()

            h = (1 - zt)*nt + zt*h
            ys.append(h)

        return F.stack(ys, axis=0).transpose(order=(1, 0, 2))


class LSTM(RNNModule):
    def __init__(self, in_size: int, hidden_size: int):
        super().__init__()

        self.hidden_size = hidden_size
        self.p_size = 4*hidden_size

        std = 1/np.sqrt(hidden_size)
        self.w_xh = Parameter.uniform(in_size, self.p_size, lo=-std, hi=std)
        self.b_xh = Parameter.uniform(self.p_size, lo=-std, hi=std)

        self.w_hh = Parameter.uniform(hidden_size, self.p_size, lo=-std, hi=std)
        self.b_hh = Parameter.uniform(self.p_size, lo=-std, hi=std)

    def forward(self, x: Parameter) -> Parameter:
        x = x.transpose(order=(1, 0, 2))
        T, N, _ = x.shape

        H = self.hidden_size
        h, c = Parameter.zeros(N, H, requires_grad=False), Parameter.zeros(N, H, requires_grad=False)

        ys: List[Parameter] = []
        for t in range(T):
            xh = F.addmm(x[t, ...], self.b_xh, self.w_xh)
            hh = F.addmm(h, self.b_hh, self.w_hh)

            it = F.sigmoid(xh[:, :H] + hh[:, :H])
            ft = F.sigmoid(xh[:, H:2*H] + hh[:, H:2*H])
            gt = (xh[:, 2*H:3*H] + hh[:, 2*H:3*H]).tanh()
            ot = F.sigmoid(xh[:, 3*H:] + hh[:, 3*H:])

            c = ft*c + it*gt
            h = ot*c.tanh()

            ys.append(h)

        return F.stack(ys, axis=0).transpose(order=(1, 0, 2))


class Bidirectional(RNNModule):
    def __init__(self, fw_layer: RNNModule, bw_layer: RNNModule, seq_axis: int = 1):
        super().__init__()

        self.fw_layer, self.bw_layer = fw_layer, bw_layer
        self.seq_axis = seq_axis

    def forward(self, x: Parameter) -> Parameter:
        x_rev = F.reverse(x, axis=self.seq_axis)
        return F.stack([self.fw_layer(x), self.bw_layer(x_rev)])


class MultiHeadAttention(Module):
    def __init__(self, embedding_dim: int, n_heads: int):
        super().__init__()

        self.e, self.h = embedding_dim, n_heads

        self.kqv = [
            Linear(self.e, self.e*self.h, bias=False),
            Linear(self.e, self.e*self.h, bias=False),
            Linear(self.e, self.e*self.h, bias=False),
        ]

        self.w = Linear(embedding_dim*n_heads, embedding_dim)

    def forward(self, x: Parameter) -> Parameter:
        N, T, K = x.shape
        int_shape, n_shape = (N, T, self.h, K), (N*self.h, T, K)

        # Should maybe implement a .view in the future...
        k, q, v = [l(x).reshape(*int_shape).transpose(order=(0, 2, 1, 3)).reshape(*n_shape) for l in self.kqv]

        w_p = q@k.transpose(order=(0, 2, 1))
        w = F.softmax(w_p/np.sqrt(self.e), axis=2)

        y = (w@v).reshape(N, self.h, T, K).transpose(order=(0, 2, 1, 3)).reshape(N, T, self.h*K)
        return self.w(y)


class Embedding(Module):
    def __init__(self, n_embeddings: int, embedding_dim: int):
        super().__init__()

        self.w = Parameter.init(n_embeddings, embedding_dim)

    # Takes indices
    def forward(self, x: Parameter) -> Parameter:
        N, _ = x.shape
        return F.stack([self.w[x[i, :], :] for i in range(N)], axis=0)


class Transformer(Module):
    def __init__(self, embedding_dim: int, n_heads: int, linear_dim: int,
                 dropout_rate: float = 0.25):
        self.att = MultiHeadAttention(embedding_dim, n_heads)
        self.ff = Sequential([
            Linear(embedding_dim, linear_dim),
            F.relu,
            Linear(linear_dim, embedding_dim),
        ])
        self.lnorm1, self.lnorm2 = LayerNorm(-1), LayerNorm(-1)

        self.p = dropout_rate

    def forward(self, x: Parameter) -> Parameter:
        x = self.lnorm1(x + DropOut(self.p)(self.att(x)))
        return self.lnorm2(x + DropOut(self.p)(self.ff(x)))


class DropOut(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()

        assert p > 0 and p < 1
        self.p, self.compensate = p, 1/(1 - p)

    def forward(self, x: Parameter) -> Parameter:
        if self._train:
            mask = np.random.binomial(1, 1 - self.p, size=x.shape)
            return x*mask*self.compensate

        return x


class LayerNorm(Module):
    def __init__(self, axis: int, eps: float = 1e-5):
        self.axis = axis
        self.eps = eps

    def forward(self, x: Parameter) -> Parameter:
        d = x - x.mean(axis=self.axis, keepdim=True)

        sigma = (d**2).mean(axis=self.axis, keepdim=True)
        return d/(sigma + self.eps).sqrt()


class Sequential(Module):
    def __init__(self, layers: Sequence[Callable[[Parameter], Parameter]]):
        super().__init__()

        self.layers = layers

    def forward(self, x: Parameter) -> Parameter:
        for l in self.layers:
            x = l(x)

        return x
