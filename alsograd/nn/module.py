from typing import Set, Any, Sequence, Callable, Generator

from alsograd.core import Parameter
from alsograd.utils import plural


# Very simple PyTorch like module implementation
class Module:
    def __init__(self) -> None:
        self._parameters: Set[str] = set()
        self._modules: Set[str] = set()

    def __setattr__(self, key: str, value: Any) -> None:
        # Check whether parameters in list (i.e. sequential)
        is_seq = isinstance(value, (tuple, list))
        if is_seq:
            value = list(filter(lambda v: isinstance(v, (Parameter, Module)), value))
            assert all(type(v) == type(value[0]) for v in value),\
                f"Not all types in sequence are the same for attr. {key}"

        value_t = value[0] if is_seq else value
        if isinstance(value_t, Parameter):
            self._parameters.add(key)
        elif isinstance(value_t, Module):
            self._modules.add(key)

        object.__setattr__(self, key, value)

    # Generator to deal with references
    def parameters(self) -> Generator[Parameter, None, None]:
        for k in self._parameters:
            yield from plural(self.__dict__[k])

        for k in self._modules:
            for m in plural(self.__dict__[k]):
                yield from m.parameters()

    def __len__(self) -> int:
        return len(list(self.parameters()))

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    forward: Callable[..., Parameter]

    def __call__(self, *args) -> Parameter:
        return self.forward(*args)
