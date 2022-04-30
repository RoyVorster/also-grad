from alsograd.core import Parameter
from alsograd.nn.module import Module


class Linear(Module):
    def __init__(self, in_neurons: int, out_neurons: int):
        super().__init__()
        self.w = Parameter.init(in_neurons, out_neurons)
        self.b = Parameter.init(out_neurons)

    def forward(self, x: Parameter) -> Parameter:
        return x@self.w + self.b
