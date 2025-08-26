import numpy as np

from tensor import Tensor

def sum(t: Tensor, *args, **kwargs):
    out = np.sum(t.data, *args, **kwargs)
    out = Tensor(out, _children=(t, ))
    
    def backward():
        res = np.ones_like(t.data) * out.grad
    out._backward = backward
    
    return out