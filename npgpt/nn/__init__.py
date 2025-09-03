"""
Neural Network modules for npgpt.

This module provides high-level neural network components for building transformers
and other architectures, similar to PyTorch's nn module.
"""

from .module import Module
from .attention import MultiHeadAttention
# from .linear import Linear
# from .embedding import Embedding
# from .sequential import Sequential

__all__ = [
    "Module",
    "MultiHeadAttention",
    # "Linear", 
    # "Embedding",
    # "Sequential",
]
