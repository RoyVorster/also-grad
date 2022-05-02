from __future__ import annotations

from typing import Dict, List, Any, Callable

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

        object.__setattr__(self, name, value)

    @property
    def parameters(self) -> List[Parameter]:
        p = list(self._parameters.values())
        for m in self._modules.values():
            p += m.parameters

        return p

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.zero_grad()

    forward: Callable[..., Parameter]

    def __call__(self, *args) -> Parameter:
        return self.forward(*args)
