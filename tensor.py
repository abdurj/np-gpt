import numpy as np

class Tensor:
    def __init__(self, *args, _children=(), _op='', **kwargs):
        self.data = np.array(*args, **kwargs)
        self.grad = np.zeros_like(self.data)
        
        # Autograd metadata
        self.op = _op
        self._prev = set(_children)
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op='+')
        
        # gradients propagate equally through addition
        # a + b = c
        # da/dl = da/dc * dc/dl
        """
            Gradients propagate equally through addition
            
            c = a + b
            da/dl   = da/dc * dc/dl
                    = 1 * dc
                    = dc
        """
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        
        return out
        
    def __repr__(self):
        return f"Tensor(\n{self.data}\n)"
    
    def zero_grad(self):
        self.grad *= 0
        
    def backward(self, grad=None):
        # TODO: collect children and top sort
        # work backwards and apply backward on children
        if grad is None:
            if self.data.size != 1:
                raise ValueError("Grad must be specified for non-scalar tensors")
            grad = np.ones_like(self.data)
        self.grad += grad
        
    # forward numpy semantics
    def __getitem__(self, key):
        result = self.data[key]
        if np.isscalar(result):
            return result
        return Tensor(result)
    
    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            self.data[key] = value.data
        else:
            self.data[key] = value
        