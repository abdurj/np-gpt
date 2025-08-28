import numpy as np


def np_allclose(a, b, rtol=1e-5, atol=1e-7):
    return np.allclose(a, b, rtol=rtol, atol=atol)

# Broadcasting works by replicating the input data over some
# dimensions. This means that for some value a inside the original tensor
# it contributes multiple times to the gradient
# as a result we need to figure out where it was applied in the forward pass
# and reduce all of those back to the original spot in the gradient
def reduce_to_shape(grad, dims):
    if len(dims) == 0:
        return grad.sum()
    
    extra_dims = len(grad.shape) - len(dims)
    for i in range(extra_dims):
        grad = grad.sum(axis=0)
    
    # shouldve collapsed all extra dims, now unbroadcast dims
    for i in range(len(dims)):
        if dims[i] == 1 and grad.shape[i] > 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad
