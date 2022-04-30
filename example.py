import alsograd.core as ag
from alsograd.nn.module import Module
from alsograd.nn.layers import Linear
from alsograd.nn.loss import MSE
from alsograd.nn.activations import ReLU


class Net(Module):
    def __init__(self):
        super().__init__()

        # Layers
        self.l1 = Linear(10, 128)
        self.l2 = Linear(128, 128)
        self.l3 = Linear(128, 2)

        # Non-linearity
        self.act = ReLU()

    def forward(self, x):
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        return self.l3(x)


# Simple forward pass
model = Net()
x = ag.Parameter.init(1, 10)
y = model(x)

y.backward()
print(model.l1.w.grad)
