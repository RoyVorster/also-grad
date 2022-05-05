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


class Adam(Optimizer):
    def __init__(self, model: Module, learning_rate: float = 1e-3, momentum: float = 0.9,
                 beta: float = 0.99, beta_inv: Optional[float] = None, bias_correction: bool = True,
                 delta: float = 1e-4) -> None:
        super().__init__(model)

        self.learning_rate = learning_rate

        # SGD
        self.momentum = momentum
        assert self.momentum >= 0 and self.momentum <= 1,\
            f"Incorrect momentum variable ({momentum})for RMSProp step"

        # RMSProp
        self.beta, self.beta_inv = beta, (1 - beta) if beta_inv is None else beta_inv
        assert self.beta == 0 or (self.beta == 1 and self.beta_inv == 1) or (self.beta <= 1 and beta_inv is None),\
            f"Incorrect beta ({self.beta}) variable for RMSProp step"

        self.delta = delta
        self.bias_correction = bias_correction

        # State
        self.v: List[Optional[np.ndarray]] = [None]*len(self.model)
        self.m: List[Optional[np.ndarray]] = [None]*len(self.model)

        self.t: int = 1

    def parameter_step(self, index: int, p: Parameter) -> None:
        if not p.grad:
            return

        g = p.grad.data

        use_rms = self.beta > 0 and self.beta_inv > 0
        if use_rms:
            v_prev = self.v[index]
            v_new = self.beta*(np.zeros_like(g) if v_prev is None else v_prev) + self.beta_inv*g**2

            if self.bias_correction:
                v_new = v_new/(1 - self.beta**self.t)

            self.v[index] = v_new

        m_new = g
        if self.momentum > 0:
            m_prev = self.m[index]
            m_new = self.momentum*(np.zeros_like(g) if m_prev is None else m_prev) + (1 - self.momentum)*g

            if self.bias_correction:
                m_new = m_new/(1 - self.momentum**self.t)

            self.m[index] = m_new

        update = self.learning_rate*m_new
        if use_rms:
            update /= (np.sqrt(v_new) + self.delta)

        p.data -= update

        self.t += 1


# All optimizers are just subsets of Adam
SGD = partial(Adam, beta=0, beta_inv=0, bias_correction=False)
RMSProp = partial(Adam, momentum=0, bias_correction=False)
AdaGrad = partial(Adam, beta=1, beta_inv=1, momentum=0, bias_correction=False)
