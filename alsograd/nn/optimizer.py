import numpy as np
from typing import Callable, Optional, List
from functools import partial

from alsograd.core import Parameter
from alsograd.nn.module import Module


class Optimizer:
    def __init__(self, model: Module) -> None:
        self.model = model

    parameter_step: Callable[[int, Parameter], None]

    def zero_grad(self) -> None:
        self.model.zero_grad()

    def step(self):
        for i, p in enumerate(self.model.parameters()):
            self.parameter_step(i, p)


class SGD(Optimizer):
    def __init__(self, model: Module, learning_rate: float = 1e-4, momentum: float = 0) -> None:
        super().__init__(model)

        self.learning_rate = learning_rate
        self.momentum = momentum

        # State
        self.grad_prev: List[Optional[np.ndarray]] = [None]*len(self.model)

    def parameter_step(self, index: int, p: Parameter) -> None:
        if not p.grad:
            return

        g = p.grad.data
        if self.momentum > 0:
            g_prev = self.grad_prev[index]
            g = g*(1 - self.momentum) + self.momentum*(g if g_prev is None else g_prev)

            self.grad_prev[index] = g

        p.data -= self.learning_rate*g


class RMSProp(Optimizer):
    def __init__(self, model: Module, learning_rate: float = 1e-4, alpha: float = 0.9, delta: float = 1e-5) -> None:
        super().__init__(model)

        self.learning_rate = learning_rate
        self.alpha = alpha
        self.delta = delta

        # State
        self.g: List[Optional[np.ndarray]] = [None]*len(self.model)

    def parameter_step(self, index: int, p: Parameter) -> None:
        if not p.grad:
            return

        g = p.grad.data

        g_prev = self.g[index]
        g_new = self.alpha*(g_prev if g_prev is not None else np.zeros_like(g)) + (1 - self.alpha)*g**2

        p.data -= self.learning_rate*g/(np.sqrt(g_new) + self.delta)
        self.g[index] = g_new


AdaGrad = partial(RMSProp, alpha=0)
