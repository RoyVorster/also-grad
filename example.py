from alsograd.core import *
from alsograd.operations import *

x = Parameter.ones(10, 1)
yy = Parameter(5)
y = Parameter.ones(10, 1)*yy

z = (x + y).sum()
z.backward()

print(z, z.grad)
print(yy.grad, y.grad, x.grad)
