import numpy as np
import torch
import npgpt
from npgpt.utils import np_allclose

"""
Tests for nonlinearity operations (ReLU, tanh, sigmoid, etc.) comparing against PyTorch.

Run: pytest tests/test_nonlinearity.py -sv
"""

class TestReLU:
    """Tests for ReLU activation function"""
    
    def test_relu_forward_backward(self):
        """Test ReLU forward and backward pass with mixed values"""
        X = np.array([[-1.0, 0.0, 1.0, 2.0],
                      [3.0, -0.5, 0.5, -3.0]], dtype=np.float32)
        
        # Torch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = torch.relu(tx)
        loss = torch_out.sum()
        loss.backward()
        
        # npgpt
        nx = npgpt.Tensor(X)
        np_out = npgpt.relu(nx)
        np_loss = np_out.sum()
        np_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)
    
    def test_relu_chain_operations(self):
        """Test ReLU in a chain of operations to verify gradient propagation"""
        X = np.array([[-1.0, 2.0, -3.0],
                      [4.0, -5.0, 6.0]], dtype=np.float32)
        W = np.array([[1.0, -1.0],
                      [0.5, 0.5],
                      [-1.0, 1.0]], dtype=np.float32)
        
        # Torch: X -> ReLU -> matmul -> sum
        tx = torch.tensor(X, requires_grad=True)
        tw = torch.tensor(W, requires_grad=True)
        torch_relu = torch.relu(tx)
        torch_out = torch_relu @ tw
        loss = torch_out.sum()
        loss.backward()
        
        # npgpt: X -> ReLU -> matmul -> sum
        nx = npgpt.Tensor(X)
        nw = npgpt.Tensor(W)
        np_relu = npgpt.relu(nx)
        np_out = np_relu @ nw
        np_loss = np_out.sum()
        np_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)
        assert np_allclose(tw.grad.numpy(), nw.grad)


    def test_manual_relu_backward(self):
        """Manually verify ReLU gradients by direct backward call"""
        np.random.seed(15)
        
        X = np.random.uniform(-1, 1, (2,3))
        Y = np.random.uniform(-1, 1, (2,3))
        
        nx = npgpt.Tensor(X)
        ny = npgpt.Tensor(Y)
        
        # Forward Pass
        nxy = nx * ny # shape (2,3)
        nrelu = npgpt.relu(nxy) # shape (2,3)
        nout = nrelu.sum() # shape (1,)
        # Backwards Pass
        nout.backward()
        # Manual Backward Pass
        
        dout = np.ones_like(nout.data) # shape (1,)
        assert np_allclose(dout, nout.grad)
        
        # relu used by taking sum
        drelu = np.ones_like(nrelu.data) * dout # shape (2,3)
        assert np_allclose(drelu, nrelu.grad)
        
        # nxy used by taking relu
        dnxy = nxy.data > 0 * drelu # shape (2, 3)
        assert np_allclose(dnxy, nxy.grad)
    
        dnx = ny.data * dnxy
        dny = nx.data * dnxy
        assert np_allclose(dnx, nx.grad)
        assert np_allclose(dny, ny.grad)
        
        
         

        
        