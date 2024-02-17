import numpy as np
from Step01 import *

"""f = Square()
a = f(Variable(np.array(1.0)))
f.generation = 0
f_p = PrioritizedFunction(f.generation, f)

print(id(f.generation))
print(id(f_p.priority))

f_p_i = PrioritizedFunction(-1 * f.generation, f)

print(id(f_p_i.priority))
print("a")

*the codes up above are test of intriguing phenomenon
"""

x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()

print(y.data)
print(x.grad)
