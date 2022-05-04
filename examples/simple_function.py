import numpy as np
import alsograd.core as ag

x = ag.Parameter(np.linspace(1, 20, 1000))
y = (x.sin() + 1).exp()/x.sqrt()

y.backward()  # Populate gradients
assert x.grad is not None

# Plot
import matplotlib.pyplot as plt
plt.plot(x.data, y.data, label="f")
plt.plot(x.data, x.grad.data, label="df")

plt.grid()
plt.legend()

plt.show()
