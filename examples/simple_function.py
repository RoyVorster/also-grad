import numpy as np
import alsograd.core as ag

x = ag.Parameter(np.linspace(0, 20, 1000))
y = x.sin()

y.backward()  # Populate gradients
assert x.grad is not None

# Plot
import matplotlib.pyplot as plt
plt.plot(x.data, y.data, label="sin")
plt.plot(x.data, x.grad.data, label="grad (cos)")

plt.grid()
plt.legend()

plt.show()
