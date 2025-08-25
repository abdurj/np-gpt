import numpy as np

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

    # Support right-add as well (e.g., 2 + Tensor)
    __radd__ = __add__

    def sum(self):
        """Sum all elements to a scalar Tensor."""
        out = Tensor(self.data.sum(), _children=(self,), _op='sum')
        
        def backward():
            # d(sum)/d(self) = 1, broadcasted to self.shape
            self.grad += np.ones_like(self.data) * out.grad
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
