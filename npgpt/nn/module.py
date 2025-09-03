"""
Base module class for all neural network modules in npgpt.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..tensor import Tensor


class Module(ABC):
    """Base class for all neural network modules.
    
    Similar to PyTorch's nn.Module, this provides the foundation for
    all neural network components with parameter management and
    forward pass definition.
    """
    
    def __init__(self):
        self._parameters: Dict[str, Tensor] = {}
        self._modules: Dict[str, 'Module'] = {}
        self.training: bool = True
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Define the forward computation performed at every call."""
        pass
    
    def __call__(self, *args, **kwargs) -> Tensor:
        """Make the module callable, delegating to forward()."""
        return self.forward(*args, **kwargs)
    
    def parameters(self) -> List[Tensor]:
        """Return a list of all parameters in this module and its submodules."""
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params
    
    def named_parameters(self) -> List[tuple[str, Tensor]]:
        """Return a list of (name, parameter) tuples."""
        params = [(name, param) for name, param in self._parameters.items()]
        for mod_name, module in self._modules.items():
            for param_name, param in module.named_parameters():
                params.append((f"{mod_name}.{param_name}", param))
        return params
    
    def zero_grad(self):
        """Zero out gradients of all parameters."""
        for param in self.parameters():
            param.grad = np.zeros_like(param.grad) if param.grad is not None else None
    
    def train(self, mode: bool = True):
        """Set the module in training mode."""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self):
        """Set the module in evaluation mode."""
        return self.train(False)
    
    def add_parameter(self, name: str, param: Tensor):
        """Add a parameter to the module."""
        self._parameters[name] = param
    
    def add_module(self, name: str, module: 'Module'):
        """Add a child module."""
        self._modules[name] = module
    
    def __repr__(self):
        """Return a string representation of the module."""
        return f"{self.__class__.__name__}()"
