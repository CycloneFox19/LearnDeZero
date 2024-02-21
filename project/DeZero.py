import numpy as np
from Step01 import *

x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
t = add(x0, x1)
y = add(x0, t)
y.backward()

print(y.grad, t.grad)

print(y.grad, t.grad)
print(x0.grad, x1.grad)