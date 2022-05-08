import numpy as np

import alsograd.core as ag
import alsograd.visualization as viz

x = ag.Parameter(np.linspace(1, 20, 1000))
y = (x.sin() + 1).exp()/x.sqrt()

y.backward()  # Populate gradients
assert x.grad is not None

x.label, y.label = "x", "y"
viz.create_graph(y, backward_order=False, render=True)

# Plot
import matplotlib.pyplot as plt
plt.plot(x.data, y.data, label="f")
plt.plot(x.data, x.grad.data, label="df")

plt.grid()
plt.legend()

plt.show()
