from typing import Callable

from alsograd.core import Parameter
from alsograd.nn.module import Module


class Optimizer:
    def __init__(self, model: Module) -> None:
        self.model = model

    parameter_step: Callable[[Parameter], None]

    def zero_grad(self) -> None:
        self.model.zero_grad()

    def step(self):
        for p in self.model.parameters():
            self.parameter_step(p)


class SGD(Optimizer):
    def __init__(self, model: Module, learning_rate: float) -> None:
        super().__init__(model)
        self.learning_rate = learning_rate

    def parameter_step(self, p: Parameter) -> None:
        if p.grad:
            p.data -= self.learning_rate*p.grad.data
