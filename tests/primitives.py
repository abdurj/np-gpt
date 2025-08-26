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
    
def test_mul_forward_backward():
    X = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    Y = np.array([3.0, 2.0, 1.0], dtype=np.float32)

    # Tensor * Tensor
    torch_x = torch.tensor(X, requires_grad=True)
    torch_y = torch.tensor(Y, requires_grad=True)
    torch_out = torch_x * torch_y
    # reduction
    z_t = torch_out.sum()
    z_t.backward()

    np_x = npgpt.Tensor(X)
    np_y = npgpt.Tensor(Y)
    np_out = np_x * np_y
    # reduction
    z_n = np_out.sum()
    z_n.backward()

    assert np_allclose(torch_out.detach().numpy(), np_out.data)
    assert np_allclose(torch_x.grad.numpy(), np_x.grad)
    assert np_allclose(torch_y.grad.numpy(), np_y.grad)

    # Scalar * Tensor
    torch_x2 = torch.tensor(X, requires_grad=True)
    torch_out2 = 2.0 * torch_x2
    z_t2 = torch_out2.sum()
    z_t2.backward()

    np_x2 = npgpt.Tensor(X)
    np_out2 = 2.0 * np_x2
    z_n2 = np_out2.sum()
    z_n2.backward()

    assert np_allclose(torch_out2.detach().numpy(), np_out2.data)
    assert np_allclose(torch_x2.grad.numpy(), np_x2.grad)


def test_zero_grad():
    X = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    np_x = npgpt.Tensor(X)
    np_y = npgpt.Tensor(X)
    z = (np_x + np_y).sum()
    z.backward()
    assert np.all(np_x.grad != 0)
    
    np_x.zero_grad()
    assert np.all(np_x.grad == 0)

def test_add_broadcast_forward_backward():
    X = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]], dtype=np.float32)      # (2,3)
    b = np.array([10.0, 20.0, 30.0], dtype=np.float32)     # (3,) -> broadcast to (2,3)

    # Torch
    tx = torch.tensor(X, requires_grad=True)
    tb = torch.tensor(b, requires_grad=True)
    torch_out = tx + tb
    zt = torch_out.sum()
    zt.backward()

    # npgpt
    nx = npgpt.Tensor(X)
    nb = npgpt.Tensor(b)
    np_out = nx + nb
    zn = np_out.sum()
    zn.backward()

    assert np_allclose(torch_out.detach().numpy(), np_out.data)
    assert np_allclose(tx.grad.numpy(), nx.grad)
    assert np_allclose(tb.grad.numpy(), nb.grad)


def test_mul_broadcast_forward_backward():
    X = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]], dtype=np.float32)      # (2,3)
    w = np.array([0.5, -1.0, 2.0], dtype=np.float32)       # (3,) -> broadcast to (2,3)

    # Torch
    tx = torch.tensor(X, requires_grad=True)
    tw = torch.tensor(w, requires_grad=True)
    torch_out = tx * tw
    zt = torch_out.sum()
    zt.backward()

    # npgpt
    nx = npgpt.Tensor(X)
    nw = npgpt.Tensor(w)
    np_out = nx * nw
    zn = np_out.sum()
    zn.backward()

    assert np_allclose(torch_out.detach().numpy(), np_out.data)
    assert np_allclose(tx.grad.numpy(), nx.grad)
    assert np_allclose(tw.grad.numpy(), nw.grad)
