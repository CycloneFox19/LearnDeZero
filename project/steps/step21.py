import numpy as np
import weakref
from dataclasses import dataclass, field
from typing import Any
import heapq
import contextlib
import unittest

class Config:
    enable_backprop = True
    

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)
        

def no_grad():
    return using_config("enable_backprop", False)
    

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


@dataclass(order=True)
class PrioritizedFunction:
    priority : int
    item : Any=field(compare=False)
    

class Variable:
    __array_priority__ = 200
    
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported.".format(type(data)))
            self.data = data
            self.name = name
            self.grad = None
            self.creator = None
            self.generation = 0
            
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + ' ' * 9)
        return "variable(" + p + ')'
        
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation
    
    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
            
        funcs = []
        seen_set = set()
        
        def add_func(f):
            if f not in seen_set:
                f_p = PrioritizedFunction(f.generation, f) # warning : priority of f_p is exactly the instance of f.generation
                heapq.heappush(funcs, f_p)
                seen_set.add(f)
                
        add_func(self.creator)
        
        while funcs:
            f = heapq.heappop(funcs).item
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
                
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx # copy construction
                    
                if x.creator is not None:
                    add_func(x.creator)
                    
                if not retain_grad:
                    for y in f.outputs:
                        y().grad = None
                        
    def clear_grad(self):
        self.grad = None
        
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
        

class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = min([x.generation for x in inputs]) # generation is now negative number. decreases as calculation proceeds
            for output in outputs:
                output.set_creator(self)
            
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
            
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()
    

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx
    

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx
    

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy
    

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy
    

class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y
    
    def backward(self, gy):
        return gy, -gy
    

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1
    

class Pow(Function):
    def __init__(self, c):
        self.c = c
        
    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx
    

# functions

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

def neg(x):
    return Neg()(x)

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

def pow(x, c):
    return Pow(c)(x)

Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__add__ = add
Variable.__radd__ = add
Variable.__neg__ = neg
Variable.__sub__ = sub
Variable.__rsub__ = rsub # subtraction isn't commutative, therefore we need different function
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv
Variable.__pow__ = pow
