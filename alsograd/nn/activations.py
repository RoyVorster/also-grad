import numpy as np

from alsograd.core import Operation


class ReLU(Operation):
    def forward(self, a: np.ndarray) -> np.ndarray:
        self.add_to_cache(a)
        return np.maximum(a, 0)

    def backward(self, g: np.ndarray) -> np.ndarray:
        a, = self.cache
        return g*(a >= 0)
