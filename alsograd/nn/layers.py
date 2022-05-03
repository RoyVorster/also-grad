from alsograd.core import Parameter
from alsograd.nn.module import Module


class Linear(Module):
    def __init__(self, in_neurons: int, out_neurons: int):
        super().__init__()

        self.w = Parameter.init(in_neurons, out_neurons)
        self.b = Parameter.zeros(out_neurons)

    def forward(self, x: Parameter) -> Parameter:
        b_shape = [1]*(x.ndim - 1) + [-1]  # Batch reshape
        return x@self.w + self.b.reshape(*b_shape)
