import numpy as np
import torch
import torch.nn.functional as F
import npgpt
import pytest
from npgpt.utils import np_allclose

"""
Tests for softmax operation comparing against PyTorch.

Run: pytest tests/test_softmax.py -sv
"""

class TestSoftmax:
    """All softmax-related tests grouped in a class"""

    def test_default_axis(self):
        """Test softmax() with default axis=-1"""
        X = np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]], dtype=np.float32)  # (2,3)

        # PyTorch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = F.softmax(tx, dim=-1)
        loss = torch_out.sum()
        loss.backward()

        # npgpt
        nx = npgpt.Tensor(X)
        np_out = npgpt.softmax(nx)  # default axis=-1
        np_loss = np_out.sum()
        np_loss.backward()

        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)

    def test_axis_0(self):
        """Test softmax(axis=0) - softmax over rows"""
        X = np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]], dtype=np.float32)  # (2,3)

        # PyTorch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = F.softmax(tx, dim=0)  # softmax over rows
        loss = torch_out.sum()
        loss.backward()

        # npgpt
        nx = npgpt.Tensor(X)
        np_out = npgpt.softmax(nx, axis=0)  # softmax over rows
        np_loss = np_out.sum()
        np_loss.backward()

        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)

    def test_axis_1(self):
        """Test softmax(axis=1) - softmax over columns"""
        X = np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]], dtype=np.float32)  # (2,3)

        # PyTorch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = F.softmax(tx, dim=1)  # softmax over columns
        loss = torch_out.sum()
        loss.backward()

        # npgpt
        nx = npgpt.Tensor(X)
        np_out = npgpt.softmax(nx, axis=1)  # softmax over columns
        np_loss = np_out.sum()
        np_loss.backward()

        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)

    def test_negative_axis(self):
        """Test softmax with negative axis"""
        X = np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]], dtype=np.float32)  # (2,3)

        # PyTorch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = F.softmax(tx, dim=-1)  # same as dim=1
        loss = torch_out.sum()
        loss.backward()

        # npgpt
        nx = npgpt.Tensor(X)
        np_out = npgpt.softmax(nx, axis=-1)  # same as axis=1
        np_loss = np_out.sum()
        np_loss.backward()

        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)

    def test_3d_tensor(self):
        """Test softmax on 3D tensor"""
        X = np.array([[[1.0, 2.0],
                      [3.0, 4.0]],
                     [[5.0, 6.0],
                      [7.0, 8.0]]], dtype=np.float32)  # (2,2,2)

        # PyTorch - softmax over last dimension
        tx = torch.tensor(X, requires_grad=True)
        torch_out = F.softmax(tx, dim=-1)  # softmax over last dim
        loss = torch_out.sum()
        loss.backward()

        # npgpt
        nx = npgpt.Tensor(X)
        np_out = npgpt.softmax(nx, axis=-1)  # softmax over last dim
        np_loss = np_out.sum()
        np_loss.backward()

        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)

    def test_numerical_stability(self):
        """Test softmax numerical stability with large values"""
        X = np.array([[100.0, 101.0, 102.0],
                     [1000.0, 1001.0, 1002.0]], dtype=np.float32)

        # PyTorch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = F.softmax(tx, dim=-1)
        loss = torch_out.sum()
        loss.backward()

        # npgpt
        nx = npgpt.Tensor(X)
        np_out = npgpt.softmax(nx, axis=-1)
        np_loss = np_out.sum()
        np_loss.backward()

        # Check no NaN or Inf values
        assert np.all(np.isfinite(np_out.data))
        assert np.all(np.isfinite(nx.grad))

        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)

    def test_small_values(self):
        """Test softmax with very small values"""
        X = np.array([[-100.0, -101.0, -102.0],
                     [-1000.0, -1001.0, -1002.0]], dtype=np.float32)

        # PyTorch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = F.softmax(tx, dim=-1)
        loss = torch_out.sum()
        loss.backward()

        # npgpt
        nx = npgpt.Tensor(X)
        np_out = npgpt.softmax(nx, axis=-1)
        np_loss = np_out.sum()
        np_loss.backward()

        # Check no NaN or Inf values
        assert np.all(np.isfinite(np_out.data))
        assert np.all(np.isfinite(nx.grad))

        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)

    def test_single_element(self):
        """Test softmax with single element (edge case)"""
        X = np.array([[5.0]], dtype=np.float32)  # (1,1)

        # PyTorch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = F.softmax(tx, dim=-1)
        loss = torch_out.sum()
        loss.backward()

        # npgpt
        nx = npgpt.Tensor(X)
        np_out = npgpt.softmax(nx, axis=-1)
        np_loss = np_out.sum()
        np_loss.backward()

        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)

    def test_uniform_values(self):
        """Test softmax with uniform values (should give uniform probabilities)"""
        X = np.array([[2.0, 2.0, 2.0],
                     [5.0, 5.0, 5.0]], dtype=np.float32)

        # PyTorch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = F.softmax(tx, dim=-1)
        loss = torch_out.sum()
        loss.backward()

        # npgpt
        nx = npgpt.Tensor(X)
        np_out = npgpt.softmax(nx, axis=-1)
        np_loss = np_out.sum()
        np_loss.backward()

        # Each row should sum to 1
        assert np.allclose(np_out.data.sum(axis=-1), 1.0)
        # Each element in a row should be 1/3
        expected = np.full_like(np_out.data, 1.0/3.0)
        assert np_allclose(np_out.data, expected)

        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)

    def test_manual_backward(self):
        """Test manual backward computation for softmax"""
        X = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # (1,3)
        axis = -1

        # npgpt
        nx = npgpt.Tensor(X)
        np_out = npgpt.softmax(nx, axis=axis)
        scalar_out = np_out.sum()
        scalar_out.backward()

        # Manual computation
        # Forward pass
        max_val = X.max(axis=axis, keepdims=True)
        exp_vals = np.exp(X - max_val)
        sum_exp = exp_vals.sum(axis=axis, keepdims=True)
        softmax_out = exp_vals / sum_exp

        # Backward pass
        dout = np.ones_like(softmax_out)  # gradient from sum()
        sum_term = np.sum(softmax_out * dout, axis=axis, keepdims=True)
        dx_expected = softmax_out * (dout - sum_term)

        assert np_allclose(softmax_out, np_out.data)
        assert np_allclose(dx_expected, nx.grad)

    def test_chain_operations(self):
        """Test softmax in a chain of operations"""
        batch_size, seq_len, vocab_size = 2, 3, 4
        
        X = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float32)
        W = np.random.randn(vocab_size, vocab_size).astype(np.float32)
        
        # PyTorch: X -> matmul -> softmax -> sum
        tx = torch.tensor(X, requires_grad=True)
        tw = torch.tensor(W, requires_grad=True)
        torch_logits = tx @ tw
        torch_probs = F.softmax(torch_logits, dim=-1)
        loss = torch_probs.sum()
        loss.backward()
        
        # npgpt: X -> matmul -> softmax -> sum
        nx = npgpt.Tensor(X)
        nw = npgpt.Tensor(W)
        np_logits = nx @ nw
        np_probs = npgpt.softmax(np_logits, axis=-1)
        np_loss = np_probs.sum()
        np_loss.backward()
        
        assert np_allclose(torch_probs.detach().numpy(), np_probs.data)
        # For chain operations, gradients can be extremely small, use relaxed tolerances
        torch_x_grad = tx.grad.numpy()
        if np.allclose(torch_x_grad, 0.0, atol=1e-10):
            # If PyTorch returns effectively zero gradients, our gradients should be very small
            assert np.all(np.abs(nx.grad) < 1e-6), f"Expected small gradients, got max: {np.abs(nx.grad).max()}"
        else:
            assert np_allclose(torch_x_grad, nx.grad, rtol=1e-3, atol=1e-6)
        assert np_allclose(tw.grad.numpy(), nw.grad, rtol=1e-4, atol=1e-6)

    def test_probability_properties(self):
        """Test that softmax output has proper probability properties"""
        X = np.random.randn(5, 10).astype(np.float32)
        
        nx = npgpt.Tensor(X)
        np_out = npgpt.softmax(nx, axis=-1)
        
        # Each row should sum to 1 (probability distribution)
        row_sums = np_out.data.sum(axis=-1)
        assert np.allclose(row_sums, 1.0, rtol=1e-6)
        
        # All values should be non-negative
        assert np.all(np_out.data >= 0.0)
        
        # All values should be <= 1
        assert np.all(np_out.data <= 1.0)
