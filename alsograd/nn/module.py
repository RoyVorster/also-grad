from typing import Set, Any, Callable, Generator

from alsograd.core import Parameter


# Very simple PyTorch like module implementation
class Module:
    def __init__(self) -> None:
        self._parameters: Set[str] = set()
        self._modules: Set[str] = set()

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            self._parameters.add(name)
        elif isinstance(value, Module):
            self._modules.add(name)

        object.__setattr__(self, name, value)

    # Generator to deal with references
    def parameters(self) -> Generator[Parameter, None, None]:
        yield from (self.__dict__[p] for p in self._parameters)

        for m in self._modules:
            yield from self.__dict__[m].parameters()

    @property
    def n_parameters(self) -> int:
        return len(list(self.parameters()))

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    forward: Callable[..., Parameter]

    def __call__(self, *args) -> Parameter:
        return self.forward(*args)
