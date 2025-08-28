import numpy as np

from .tensor import Tensor

def relu(t: Tensor):
    out = Tensor(np.maximum(0, t.data), _children=(t,), _op='ReLU')
    
    def backward():
        dout = out.grad
        mask = (t.data > 0).astype(np.float32)
        dt = mask * dout
        t.grad += dt
    
    out._backward = backward
    return out
    