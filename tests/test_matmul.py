import numpy as np
import torch
import npgpt
import pytest

"""
Tests for matrix multiplication operations comparing against PyTorch.

Run: pytest tests/matmul.py -sv
"""

def np_allclose(a, b, rtol=1e-5, atol=1e-7):
    return np.allclose(a, b, rtol=rtol, atol=atol)


class TestMatmul:
    """Tests for matrix multiplication operations"""
    
    def test_2d_matmul(self):
        """Test basic 2D matrix multiplication A @ B"""
        A = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], dtype=np.float32)  # (2, 3)
        B = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]], dtype=np.float32)  # (3, 2)
        
        # Torch
        ta = torch.tensor(A, requires_grad=True)
        tb = torch.tensor(B, requires_grad=True)
        torch_out = ta @ tb  # (2, 2)
        loss = torch_out.sum()
        loss.backward()
        
        # npgpt
        na = npgpt.Tensor(A)
        nb = npgpt.Tensor(B)
        np_out = na @ nb  # (2, 2)
        np_loss = np_out.sum()
        np_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(ta.grad.numpy(), na.grad)
        assert np_allclose(tb.grad.numpy(), nb.grad)
    
    def test_matmul_method(self):
        """Test matmul method A.matmul(B)"""
        A = np.array([[1.0, 2.0],
                      [3.0, 4.0]], dtype=np.float32)  # (2, 2)
        B = np.array([[5.0, 6.0],
                      [7.0, 8.0]], dtype=np.float32)  # (2, 2)
        
        # Torch
        ta = torch.tensor(A, requires_grad=True)
        tb = torch.tensor(B, requires_grad=True)
        torch_out = ta.matmul(tb)  # (2, 2)
        loss = torch_out.sum()
        loss.backward()
        
        # npgpt
        na = npgpt.Tensor(A)
        nb = npgpt.Tensor(B)
        np_out = na.matmul(nb)  # (2, 2)
        np_loss = np_out.sum()
        np_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(ta.grad.numpy(), na.grad)
        assert np_allclose(tb.grad.numpy(), nb.grad)
    
    def test_vector_matrix(self):
        """Test vector @ matrix (1D @ 2D)"""
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # (3,)
        M = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]], dtype=np.float32)  # (3, 2)
        
        # Torch
        tv = torch.tensor(v, requires_grad=True)
        tm = torch.tensor(M, requires_grad=True)
        torch_out = tv @ tm
        torch_out.retain_grad()
        loss = torch_out.sum()
        loss.retain_grad()
        loss.backward()
        
        # npgpt
        nv = npgpt.Tensor(v)
        nm = npgpt.Tensor(M)
        np_out = nv @ nm  # (2,)
        np_loss = np_out.sum()
        np_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tv.grad.numpy(), nv.grad)
        assert np_allclose(tm.grad.numpy(), nm.grad)
        
        # manual backprop
        dloss = torch.ones_like(loss.data)
        assert torch.allclose(dloss, loss.grad)
        dout = torch.ones_like(torch_out) * dloss
        assert torch.allclose(dout, torch_out.grad)
        dtv = dout @ tm.T
        assert torch.allclose(dtv, tv.grad)
        dtm = tv.view(tv.shape[0], 1) @ dout.view(1, -1)
        assert torch.allclose(dtm, tm.grad)

    def test_matrix_vector(self):
        """Test matrix @ vector (2D @ 1D)"""
        M = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], dtype=np.float32)  # (2, 3)
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # (3,)
        
        # Torch
        tm = torch.tensor(M, requires_grad=True)
        tv = torch.tensor(v, requires_grad=True)
        torch_out = tm @ tv  # (2,)
        loss = torch_out.sum()
        loss.backward()
        
        # npgpt
        nm = npgpt.Tensor(M)
        nv = npgpt.Tensor(v)
        np_out = nm @ nv  # (2,)
        np_loss = np_out.sum()
        np_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tm.grad.numpy(), nm.grad)
        assert np_allclose(tv.grad.numpy(), nv.grad)
    
    def test_vector_vector(self):
        """Test vector @ vector (1D @ 1D) - dot product"""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # (3,)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)  # (3,)
        
        # Torch
        ta = torch.tensor(a, requires_grad=True)
        tb = torch.tensor(b, requires_grad=True)
        torch_out = ta @ tb  # scalar
        torch_out.backward()
        
        # npgpt
        na = npgpt.Tensor(a)
        nb = npgpt.Tensor(b)
        np_out = na @ nb  # scalar
        print(np_out)
        np_out.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(ta.grad.numpy(), na.grad)
        assert np_allclose(tb.grad.numpy(), nb.grad)
    
    def test_batched_matmul(self):
        """Test batched matrix multiplication (3D @ 3D)"""
        batch_size = 2
        A = np.random.randn(batch_size, 3, 4).astype(np.float32)  # (batch, 3, 4)
        B = np.random.randn(batch_size, 4, 5).astype(np.float32)  # (batch, 4, 5)
        
        # Torch
        ta = torch.tensor(A, requires_grad=True)
        tb = torch.tensor(B, requires_grad=True)
        torch_out = ta @ tb  # (2, 3, 5)
        loss = torch_out.sum()
        loss.backward()
        
        # npgpt
        na = npgpt.Tensor(A)
        nb = npgpt.Tensor(B)
        np_out = na @ nb  # (2, 3, 5)
        np_loss = np_out.sum()
        np_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(ta.grad.numpy(), na.grad)
        assert np_allclose(tb.grad.numpy(), nb.grad)
    
    def test_broadcast_matmul(self):
        """Test broadcasting in matmul (2D @ 3D)"""
        A = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], dtype=np.float32)  # (2, 3)
        B = np.random.randn(4, 3, 5).astype(np.float32)  # (batch, 3, 5)
        
        # Torch
        ta = torch.tensor(A, requires_grad=True)
        tb = torch.tensor(B, requires_grad=True)
        torch_out = ta @ tb  # (4, 2, 5) - A broadcasts to (4, 2, 3)
        loss = torch_out.sum()
        loss.backward()
        
        # npgpt
        na = npgpt.Tensor(A)
        nb = npgpt.Tensor(B)
        np_out = na @ nb  # (4, 2, 5)
        np_loss = np_out.sum()
        np_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(ta.grad.numpy(), na.grad)
        assert np_allclose(tb.grad.numpy(), nb.grad)


class TestLinearLayer:
    """Test linear layer pattern: X @ W + b"""
    
    def test_linear_forward_backward(self):
        """Test X @ W + b pattern"""
        # Mini-batch: 2 samples, 3 features -> 4 outputs
        X = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], dtype=np.float32)  # (2, 3)
        W = np.array([[0.1, 0.2, 0.3, 0.4],
                      [0.5, 0.6, 0.7, 0.8],
                      [0.9, 1.0, 1.1, 1.2]], dtype=np.float32)  # (3, 4)
        b = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)  # (4,)
        
        # Torch
        tx = torch.tensor(X, requires_grad=True)
        tw = torch.tensor(W, requires_grad=True)
        tb = torch.tensor(b, requires_grad=True)
        torch_out = tx @ tw + tb  # (2, 4)
        loss = torch_out.sum()
        loss.backward()
        
        # npgpt
        nx = npgpt.Tensor(X)
        nw = npgpt.Tensor(W)
        nb = npgpt.Tensor(b)
        np_out = nx @ nw + nb  # (2, 4)
        np_loss = np_out.sum()
        np_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)
        assert np_allclose(tw.grad.numpy(), nw.grad)
        assert np_allclose(tb.grad.numpy(), nb.grad)
    
    def test_manual_backward_check(self):
        """Verify gradients manually for simple case"""
        # Simple case: (1,2) @ (2,1) -> (1,1)
        X = np.array([[1.0, 2.0]], dtype=np.float32)  # (1, 2)
        W = np.array([[3.0], [4.0]], dtype=np.float32)  # (2, 1)
        
        # npgpt
        nx = npgpt.Tensor(X)
        nw = npgpt.Tensor(W)
        np_out = nx @ nw  # [[11.0]] = [[1*3 + 2*4]]
        np_out.backward()
        
        # Manual gradient calculation:
        # out = X @ W, dout/dX = W.T, dout/dW = X.T
        expected_x_grad = W.T  # (1, 2) = [[3.0, 4.0]]
        expected_w_grad = X.T  # (2, 1) = [[1.0], [2.0]]
        
        assert np_allclose(np_out.data, np.array([[11.0]]))
        assert np_allclose(nx.grad, expected_x_grad)
        assert np_allclose(nw.grad, expected_w_grad)


class TestMatmulEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_incompatible_shapes_error(self):
        """Test that incompatible shapes raise appropriate errors"""
        A = npgpt.Tensor(np.random.randn(2, 3))  # (2, 3)
        B = npgpt.Tensor(np.random.randn(4, 5))  # (4, 5) - incompatible
        
        # Should raise an error (exact error type depends on implementation)
        with pytest.raises((ValueError, RuntimeError)):
            A @ B
    
    def test_single_element_matrices(self):
        """Test 1x1 matrix multiplication"""
        A = np.array([[2.0]], dtype=np.float32)  # (1, 1)
        B = np.array([[3.0]], dtype=np.float32)  # (1, 1)
        
        # Torch
        ta = torch.tensor(A, requires_grad=True)
        tb = torch.tensor(B, requires_grad=True)
        torch_out = ta @ tb  # [[6.0]]
        torch_out.backward()
        
        # npgpt
        na = npgpt.Tensor(A)
        nb = npgpt.Tensor(B)
        np_out = na @ nb  # [[6.0]]
        np_out.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(ta.grad.numpy(), na.grad)
        assert np_allclose(tb.grad.numpy(), nb.grad)
