import sys, os
sys.path.append(os.pardir)
from steps.step21 import *

x = Variable(np.array(2.0))
y1 = 3.0 / x
y2 = x / 1.5
y3 = x ** 3

print(y1)
print(y2)
print(y3)

y3.backward(True)

print(y3.grad)
print(y2.grad)
print(y1.grad)
