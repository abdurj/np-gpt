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
    
def tanh(t: Tensor):
    out = Tensor(np.tanh(t.data), _children=(t,), _op="tanh")
    
    def backward():
        dout = out.grad
        # d/dx tanh(x) = 1 - tanh^2(x)
        dt = (1 - out.data**2) * dout
        t.grad += dt
        
    out._backward = backward
    return out

def sigmoid(t: Tensor):
    out = Tensor(1 / (1 + np.exp(-t.data)), _children=(t,), _op="sigmoid")

    def backward():
        dout = out.grad
        # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        dt = out.data * (1 - out.data) * dout
        t.grad += dt

    out._backward = backward

    return out