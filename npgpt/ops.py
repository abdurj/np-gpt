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


def layernorm(t: Tensor, gamma: Tensor, beta: Tensor, eps=1e-5):
    """
    Layer normalization as described in the paper:
    https://arxiv.org/abs/1607.06450
    
    Args:
        t: Input tensor of shape (..., embed_dim)
        gamma: Scale parameter of shape (embed_dim,) 
        beta: Shift parameter of shape (embed_dim,)
        eps: Small constant for numerical stability
    """
    assert gamma.shape()[-1] == beta.shape()[-1] \
        and gamma.shape()[-1] == t.shape()[-1]
    assert len(gamma.shape()) == 1 and len(beta.shape()) == 1, \
        "gamma and beta should be 1D tensors to match PyTorch"
    
    # Get Mean and Variance over embedding dim
    # reduction over embedding dim
    mean = np.mean(t.data, axis=-1, keepdims=True)
    var = np.var(t.data, axis=-1, keepdims=True, ddof=0)
    # broadcast over embedding dim for mean and var
    norm = (t.data - mean) / np.sqrt(var + eps)
    
    # broadcast over seq len for gamma and beta
    out = gamma.data * norm + beta.data
    
    out = Tensor(out, _children=(t, gamma, beta,), _op="layernorm")
    
    def backward():
        N = t.shape()[-1]
        
        dout = out.grad
        
        # Gradients w.r.t. gamma and beta (sum over all dims except last)
        sum_axes = tuple(range(dout.ndim - 1))
        gamma.grad += (norm * dout).sum(axis=sum_axes)
        beta.grad += dout.sum(axis=sum_axes)
        
        # Gradient w.r.t. normalized values
        dnorm = gamma.data * dout
        
        # Precompute values for backward pass
        inv_std = 1.0 / np.sqrt(var + eps)
        x_centered = t.data - mean
        
        # Branch 1: Direct gradient through normalization
        # d/dx (x-mean)/sqrt(var+eps) = 1/sqrt(var+eps) * dnorm
        dt_direct = inv_std * dnorm
        t.grad += dt_direct
        
        # Branch 2: Gradient through mean
        # mean is computed as sum(x)/N, so d_mean/dx = 1/N
        # d/d_mean [(x-mean)/sqrt(var+eps)] = -1/sqrt(var+eps)
        dmean = (-inv_std * dnorm).sum(axis=-1, keepdims=True)
        dt_mean = (1.0 / N) * dmean
        t.grad += dt_mean
        
        # Branch 3: Gradient through variance
        # var is computed as sum((x-mean)^2)/N, so d_var/dx = 2*(x-mean)/N  
        # d/d_var [(x-mean)/sqrt(var+eps)] = (x-mean) * (-0.5) * (var+eps)^(-1.5)
        dvar = (x_centered * (-0.5) * (var + eps)**(-1.5) * dnorm).sum(axis=-1, keepdims=True)
        dt_var = (2.0 / N) * x_centered * dvar
        t.grad += dt_var
        
    out._backward = backward
    
    return out