import numpy as np

class Tensor:
    def __init__(self, *args, **kwargs):
        self.data = np.array(*args, **kwargs)

    def __repr__(self):
        return f"Tensor(\n{self.data}\n)"
    
