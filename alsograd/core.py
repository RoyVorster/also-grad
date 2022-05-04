from __future__ import annotations

from contextlib import contextmanager
from copy import copy
from typing import Iterator, Union, List, Tuple, Set, Any, Optional, Callable, Iterable
import numpy as np

from alsograd.utils import plural, Axis


# Top-level grad enabled flag
enable_grad = True


@contextmanager
def no_grad() -> Iterator[None]:
    global enable_grad
    enable_grad = False
    try:
        yield
    finally:
        enable_grad = True


def wrap_parameters(f):
    def wrapper(*args, **kwargs):
        args = [arg if isinstance(arg, Parameter) else Parameter(arg, requires_grad=False) for arg in args]
        return f(*args, **kwargs)

    return wrapper


# Parameters with gradients
class Parameter:
    def __init__(self, data: Union[Parameter, np.ndarray, Iterable[Any]], requires_grad=True) -> None:
        self.data: np.ndarray = data.data if isinstance(data, Parameter) else np.asarray(data)
        self.requires_grad: bool = data.requires_grad if isinstance(data, Parameter) else requires_grad

        self.grad: Optional[Parameter] = None
        self.creator: Optional[Operation] = None

    def __str__(self) -> str:
        return str(self.data)

    def detach(self) -> Parameter:
        if not self.requires_grad:
            self.creator = None
            return self

        return Parameter(self.data, requires_grad=False)

    def zero_grad(self) -> None:
        self.grad = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @classmethod
    def zeros(cls, *shape: int, **kwargs) -> Parameter:
        return cls(np.zeros(shape, dtype=np.float32), **kwargs)

    @classmethod
    def ones(cls, *shape: int, **kwargs) -> Parameter:
        return cls(np.ones(shape, dtype=np.float32), **kwargs)

    @classmethod
    def rand(cls, *shape: int, **kwargs) -> Parameter:
        return cls(np.random.rand(*shape).astype(np.float32), **kwargs)

    @classmethod
    def init(cls, *shape: int, **kwargs) -> Parameter:
        # Standard uniform initialization
        data = np.random.uniform(-1, 1, shape)/np.sqrt(np.prod(shape))
        return cls(data.astype(np.float32), **kwargs)

    def backward(self) -> None:
        if not self.requires_grad or not enable_grad:
            return

        # DFS
        nodes: List[Parameter] = []
        seen: Set[Parameter] = set()

        def dfs_(node: Parameter):
            # If can go forward, keep going
            if node not in seen and node.creator:
                for new_node in node.creator.parents:
                    dfs_(new_node)

                nodes.append(node)

        dfs_(self)

        # Go backward through the graph
        self.grad = Parameter(np.ones_like(self.data), requires_grad=False)
        for node in reversed(nodes):
            if not node.creator or not node.grad or \
                    not any(p.requires_grad for p in node.creator.parents):
                continue

            grads = node.creator.backward_(node.grad.data)
            for (parent, grad) in zip(node.creator.parents, grads):
                if grad and parent.requires_grad:
                    parent.grad = parent.grad + grad if parent.grad else grad

    # Operations
    @wrap_parameters
    def __add__(self, other) -> Parameter:
        return ops.Add()(self, other)
    __radd__ = __iadd__ = __add__

    @wrap_parameters
    def __sub__(self, other) -> Parameter:
        return ops.Sub()(self, other)
    __rsub__ = __isub__ = __sub__

    @wrap_parameters
    def __mul__(self, other) -> Parameter:
        return ops.Mul()(self, other)
    __rmul__ = __imul__ = __mul__

    @wrap_parameters
    def __truediv__(self, other) -> Parameter:
        return ops.Div()(self, other)
    __rtruediv__ = __itruediv__ = __truediv__

    @wrap_parameters
    def __pow__(self, other) -> Parameter:
        return ops.Pow()(self, other)
    __rpow__ = __ipow__ = __pow__

    @wrap_parameters
    def __matmul__(self, other) -> Parameter:
        return ops.Dot()(self, other)

    def __neg__(self) -> Parameter:
        return ops.Neg()(self)

    def __getitem__(self, key) -> Parameter:
        return ops.Slice(key)(self)

    def sum(self, axis: Axis = None, keepdims: bool = False) -> Parameter:
        return ops.reduce_op(ops.Sum(axis=axis), self, keepdims)

    def max(self, axis: Axis = None, keepdims: bool = False) -> Parameter:
        return ops.reduce_op(ops.Max(axis=axis), self, keepdims)

    def mean(self, **kwargs) -> Parameter:
        s = self.sum(**kwargs)
        return s*(np.prod(s.shape)/np.prod(self.shape))

    def reshape(self, *shape: int) -> Parameter:
        return ops.Reshape(*shape)(self)

    def ravel(self) -> Parameter:
        return self.reshape(-1)

    def transpose(self, **kwargs) -> Parameter:
        return ops.Transpose(**kwargs)(self)

    @property
    def T(self) -> Parameter:
        return self.transpose(axis=None)

    def exp(self) -> Parameter:
        return ops.Exp()(self)

    def log(self) -> Parameter:
        return ops.Log()(self)

    def sin(self) -> Parameter:
        return ops.Sin()(self)

    def cos(self) -> Parameter:
        return ops.Cos()(self)

    def sqrt(self) -> Parameter:
        return self**0.5


# Any operation on parameters
class Operation:
    forward: Callable[..., np.ndarray]
    backward: Callable[..., Union[np.ndarray, Tuple[np.ndarray, ...]]]

    def reset(self):
        self.parents, self.cache = [], []

    def add_to_cache(self, *xs: Any) -> None:
        self.cache += xs

    # Internal wrappers
    def forward_(self, *parameters: Parameter) -> Parameter:
        self.parents = parameters

        requires_grad = any(p.requires_grad for p in parameters)
        return Parameter(self.forward(*[p.data for p in parameters]), requires_grad=requires_grad)

    def backward_(self, g: np.ndarray) -> List[Parameter]:
        return [Parameter(x, requires_grad=True) for x in plural(self.backward(g))]

    def __call__(self, *parameters) -> Parameter:
        self.reset()

        output = self.forward_(*parameters)
        if enable_grad and output.requires_grad:
            output.creator = copy(self)

        return output


class UnaryOperation(Operation):
    def __init__(self, forward: Callable[[np.ndarray], np.ndarray],
                 backward: Callable[[np.ndarray], np.ndarray]) -> None:
        self.f_forward, self.f_backward = forward, backward

    def forward(self, a: np.ndarray) -> np.ndarray:
        self.add_to_cache(a)
        return self.f_forward(a)

    def backward(self, g: np.ndarray) -> np.ndarray:
        a, = self.cache
        return g*self.f_backward(a)


class ReduceOperation(Operation):
    def __init__(self, axis: Axis):
        self.axis = axis


# Circular imports
import alsograd.operations as ops
