import numpy as np
import torch
import npgpt

"""
Basic tests for npgpt primitives, comparing against PyTorch.

Run: pytest tests/primitives.py  
"""

def np_allclose(a, b, rtol=1e-5, atol=1e-7):
    return np.allclose(a, b, rtol=rtol, atol=atol)

def test_add_forward():
    X = np.array([[1.0, 2.0], 
                  [3.0, 4.0]], dtype=np.float32)
    Y = np.array([[0.5, -1.0], 
                  [1.0, 0.0]], dtype=np.float32)

    torch_x = torch.tensor(X, requires_grad=True)
    torch_y = torch.tensor(Y, requires_grad=True)
    torch_out = torch_x + torch_y

    np_x = npgpt.Tensor(X)
    np_y = npgpt.Tensor(Y)
    np_out = np_x + np_y

    assert np_allclose(torch_out.detach().numpy(), np_out.data)


def test_add_backward_sum_scalar():
    X = np.array([[1.0, 2.0], 
                  [3.0, 4.0]], dtype=np.float32)
    Y = np.array([[0.5, -1.0], 
                  [1.0, 0.0]], dtype=np.float32)

    # Torch
    torch_x = torch.tensor(X, requires_grad=True)
    torch_y = torch.tensor(Y, requires_grad=True)
    z_t = (torch_x + torch_y).sum()
    z_t.backward()

    # npgpt
    np_x = npgpt.Tensor(X)
    np_y = npgpt.Tensor(Y)
    z_n = (np_x + np_y).sum()
    z_n.backward()

    assert np_allclose(torch_x.grad.numpy(), np_x.grad)
    assert np_allclose(torch_y.grad.numpy(), np_y.grad)


def test_radd():
    X = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    torch_x = torch.tensor(X, requires_grad=True)
    torch_out = 2.0 + torch_x

    np_x = npgpt.Tensor(X)
    np_out = 2.0 + np_x

    assert np_allclose(torch_out.detach().numpy(), np_out.data)


def test_zero_grad():
    X = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    np_x = npgpt.Tensor(X)
    np_y = npgpt.Tensor(X)
    z = (np_x + np_y).sum()
    z.backward()
    assert np.all(np_x.grad != 0)
    
    np_x.zero_grad()
    assert np.all(np_x.grad == 0)
