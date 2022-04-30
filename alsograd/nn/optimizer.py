from __future__ import annotations

from typing import Callable, List

from alsograd.core import Parameter, no_grad
from alsograd.nn.module import Module


class Optimizer:
    def __init__(self, model: Module) -> None:
        self.model = model
        self.parameters: List[Parameter] = model.parameters

    parameter_step: Callable[[Parameter], None]

    def zero_grad(self) -> None:
        self.model.zero_grad()

    def step(self):
        with no_grad():
            for p in self.parameters:
                self.parameter_step(p)


class SGD(Optimizer):
    def __init__(self, model: Module, learning_rate: float) -> None:
        super().__init__(model)

        self.learning_rate = learning_rate

    def parameter_step(self, parameter: Parameter) -> None:
        if parameter.grad:
            parameter -= Parameter([self.learning_rate], requires_grad=False)*parameter.grad
