from typing import Dict, List, Any, Callable
import numpy as np

from alsograd.core import Parameter


# Very simple PyTorch like module implementation
class Module:
    def __init__(self) -> None:
        self._parameters: Dict[str, Parameter] = {}
        self._modules: Dict[str, Module] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        else:
            object.__setattr__(self, name, value)

    @property
    def parameters(self) -> List[Parameter]:
        p = list(self._parameters.values())
        for m in self._modules.values():
            p += m.parameters

        return p

    forward: Callable[..., Parameter]

    def __call__(self, *args):
        return self.forward(args)


class Linear(Module):
    def __init__(self, in_neurons: int, out_neurons: int):
        super().__init__()
        self.w = Parameter.init(in_neurons, out_neurons)
        self.b = Parameter.init(out_neurons)

    def forward(self, x: Parameter) -> Parameter:
        return x@self.w + self.b
