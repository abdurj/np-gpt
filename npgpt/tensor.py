import numpy as np
from .utils import reduce_to_shape

class Tensor:
    def __init__(self, *args, _children=(), _op='', **kwargs):
        self.data = np.array(*args, **kwargs)
        self.grad = np.zeros_like(self.data)
        
        # Autograd metadata
        self.op = _op
        self._prev = set(_children)
        self._backward = lambda: None
    
    def __repr__(self):
        return f"Tensor(\n{self.data}\n)"
    
    def shape(self):
        return self.data.shape
    
    def dtype(self):
        return self.data.dtype

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op='+')
        
        def backward():
            dout = out.grad
            self.grad += reduce_to_shape(dout.copy(), self.shape()) # * 1
            other.grad += reduce_to_shape(dout.copy(), other.shape()) # * 1 
            
        out._backward = backward
        
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other), _op='*')
        
        def backward():
            dout = out.grad
            self_grad = other.data * dout
            other_grad = self.data * dout
            
            self.grad += reduce_to_shape(self_grad, self.shape())
            other.grad += reduce_to_shape(other_grad, other.shape())
        
        out._backward = backward
        return out

    __rmul__ = __mul__

    def __sub__(self, other):
        # sub can be implemented as self + (-1 * other)
        return self + (-other)
    
    def __pow__(self, exp):
        out = Tensor(self.data ** exp, _children=(self), _op='**')
        def backward():
            dout = out.grad
            self.grad += (exp * (self.data ** (exp-1))) * dout

        out._backward = backward
        return out
    
    def __truediv__(self, other):
        # div can be implemented as a combination of other primitives
        return self * (other ** -1)
    
    def __neg__(self):
        return -1 * self
    
    def sum(self, axis=None, keepdims=False, **kwargs):
        """Sum all elements to a scalar Tensor."""
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims, **kwargs), _children=(self,), _op='sum')
        
        def backward():
            # d(sum)/d(self) = 1, broadcasted to self.shape
            dout = out.grad
            if axis is None or keepdims:
                # if keepdim == True then we can broadcast dout back to self shape
                self.grad += np.ones_like(self.data) * dout
            else:
                # restore collapsed dimension
                grad = np.expand_dims(dout, axis=axis)
                # broadcast through to collapsed dimension, since it got reduced
                # but gradient flows through all elements equally``
                grad = np.broadcast_to(grad, self.shape())
                self.grad += grad 
                
        out._backward = backward
        return out
    
    def mean(self, axis=None, keepdims=False, **kwargs):
        """Average all elements down to a scalar Tensor"""
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims, **kwargs), _children=(self), _op="mean")
        
        def backward():
            dout = out.grad
            if axis is None:
                self.grad += (np.ones_like(self.data) / self.data.size) * dout
            elif keepdims:
                # get num elements along the axis it was mean'd over
                self.grad += (np.ones_like(self.data) / self.data.shape[axis]) * dout
            else:
                # restore collapsed dimension
                grad = np.expand_dims(dout, axis=axis)
                # broadcast through to collapsed dimension, since it got reduced
                grad = np.broadcast_to(grad, self.shape())
                # get num elements along the axis it was mean'd over
                self.grad += (grad / self.data.shape[axis])

        out._backward = backward
        return out
    
    def transpose(self, axes=None):
        out = Tensor(self.data.transpose(axes), _children=(self,), _op='transpose')
        
        def backward():
            dout = out.grad
            if axes is None:
                self.grad += dout.transpose()
            else:
                # invert the axes permutation
                inv_axes = np.argsort(axes)
                self.grad += dout.transpose(inv_axes)

        out._backward = backward
        return out

    T = property(transpose)  # allow x.T as shorthand for x.transpose()
    
    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), _children=(self,), _op='reshape')
        
        def backward():
            dout = out.grad
            self.grad += dout.reshape(self.shape())
        
        out._backward = backward
        return out

    def zero_grad(self):
        self.grad *= 0
        
    def backward(self, grad=None):
        """Backpropagate gradients through the computation graph.
        If grad is None, assumes this tensor is a scalar and starts with 1.
        """

        # Simple DFS TopSort
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        # Initialize gradient at the output
        if grad is None:
            if self.data.size != 1:
                raise ValueError("Grad must be specified for non-scalar tensors")
            grad = np.ones_like(self.data)
            
        self.grad += grad
        # Traverse in reverse topological order and call each node's backward
        for v in reversed(topo):
            v._backward()
        
    # forward numpy slicing semantics
    def __getitem__(self, key):
        result = self.data[key]

        out = Tensor(result, _children=(self,), _op='slice')
        
        def backward():
            dout = out.grad
            # Create a zero array of the original shape
            grad = np.zeros_like(self.data)
            # Place dout into the correct slice
            grad[key] = dout
            self.grad += grad
        
        out._backward = backward
        
        return out
    
    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            self.data[key] = value.data
        else:
            self.data[key] = value
