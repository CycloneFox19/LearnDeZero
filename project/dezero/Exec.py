import sys, os
sys.path.append(os.pardir)
from steps.step21 import *

x = Variable(np.array(2.0))
y = 3.0 * x + 1.0
print(y)
