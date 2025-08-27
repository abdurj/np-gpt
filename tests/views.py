import numpy as np
import torch
import npgpt
import pytest

"""
Tests for tensor shape operations (slice, reshape, transpose) comparing against PyTorch.

Run: pytest tests/views.py -sv
"""

def np_allclose(a, b, rtol=1e-5, atol=1e-7):
    return np.allclose(a, b, rtol=rtol, atol=atol)


class TestSlicing:
    """Tests for tensor slicing operations"""
    
    def test_single_row(self):
        """Test x[0] - extract first row"""
        X = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], dtype=np.float32)
        
        # Torch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = tx[0]  # shape (3,)
        loss = torch_out.sum()
        loss.backward()
        
        # npgpt
        nx = npgpt.Tensor(X)
        np_out = nx[0]  # shape (3,)
        np_loss = np_out.sum()
        np_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)
    
    def test_column_slice(self):
        """Test x[:, 1:] - extract columns 1,2"""
        X = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], dtype=np.float32)
        
        # Torch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = tx[:, 1:]  # shape (2, 2)
        loss = torch_out.sum()
        loss.backward()
        
        # npgpt
        nx = npgpt.Tensor(X)
        np_out = nx[:, 1:]  # shape (2, 2)
        np_loss = np_out.sum()
        np_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)
    
    def test_single_element(self):
        """Test x[0, 1] - extract single element"""
        X = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], dtype=np.float32)
        
        # Torch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = tx[0, 1]  # scalar
        torch_out.backward()
        
        # npgpt
        nx = npgpt.Tensor(X)
        np_out = nx[0, 1]  # should be scalar or 0-d tensor
        if hasattr(np_out, 'backward'):  # if it's a Tensor
            np_out.backward()
        
        # For single element, we expect only that position to have gradient
        expected_grad = np.zeros_like(X)
        expected_grad[0, 1] = 1.0
        
        assert np_allclose(tx.grad.numpy(), expected_grad)
        if hasattr(np_out, 'data'):  # if np_out is Tensor
            assert np_allclose(torch_out.detach().numpy(), np_out.data)
            assert np_allclose(tx.grad.numpy(), nx.grad)
    
    def test_strided_slice(self):
        """Test x[::2] - every other row"""
        X = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0],
                      [7.0, 8.0]], dtype=np.float32)
        
        # Torch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = tx[::2]  # shape (2, 2) - rows 0, 2
        loss = torch_out.sum()
        loss.backward()
        
        # npgpt
        nx = npgpt.Tensor(X)
        np_out = nx[::2]  # shape (2, 2)
        np_loss = np_out.sum()
        np_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)


class TestReshape:
    """Tests for tensor reshape operations"""
    
    def test_flatten(self):
        """Test reshape to 1D"""
        X = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], dtype=np.float32)  # (2, 3)
        
        # Torch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = tx.reshape(6)  # (6,)
        loss = torch_out.sum()
        loss.backward()
        
        # npgpt
        nx = npgpt.Tensor(X)
        np_out = nx.reshape(6)  # (6,)
        np_loss = np_out.sum()
        np_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)
    
    def test_reshape_2d(self):
        """Test reshape (2,3) -> (3,2)"""
        X = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], dtype=np.float32)  # (2, 3)
        
        # Torch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = tx.reshape(3, 2)  # (3, 2)
        loss = torch_out.sum()
        loss.backward()
        
        # npgpt
        nx = npgpt.Tensor(X)
        np_out = nx.reshape(3, 2)  # (3, 2)
        np_loss = np_out.sum()
        np_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)
    
    def test_reshape_3d(self):
        """Test reshape to 3D"""
        X = np.array([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)  # (2, 4)
        
        # Torch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = tx.reshape(2, 2, 2)  # (2, 2, 2)
        loss = torch_out.sum()
        loss.backward()
        
        # npgpt
        nx = npgpt.Tensor(X)
        np_out = nx.reshape(2, 2, 2)  # (2, 2, 2)
        np_loss = np_out.sum()
        np_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)


class TestTranspose:
    """Tests for tensor transpose operations"""
    
    def test_transpose_2d(self):
        """Test basic 2D transpose"""
        X = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], dtype=np.float32)  # (2, 3)
        
        # Torch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = tx.transpose(0, 1)  # (3, 2)
        loss = torch_out.sum()
        loss.backward()
        
        # npgpt
        nx = npgpt.Tensor(X)
        np_out = nx.transpose([1, 0])  # (3, 2)
        np_loss = np_out.sum()
        np_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)
    
    def test_transpose_property(self):
        """Test .T property"""
        X = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], dtype=np.float32)  # (2, 3)
        
        # Torch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = tx.T  # (3, 2)
        loss = torch_out.sum()
        loss.backward()
        
        # npgpt
        nx = npgpt.Tensor(X)
        np_out = nx.T  # (3, 2)
        np_loss = np_out.sum()
        np_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)
    
    def test_transpose_3d(self):
        """Test 3D transpose with specific axes"""
        X = np.array([[[1.0, 2.0],
                       [3.0, 4.0]],
                      [[5.0, 6.0],
                       [7.0, 8.0]]], dtype=np.float32)  # (2, 2, 2)
        
        # Torch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = tx.permute(2, 0, 1)  # (2, 2, 2) -> (2, 2, 2) but permuted
        loss = torch_out.sum()
        loss.backward()
        
        # npgpt
        nx = npgpt.Tensor(X)
        np_out = nx.transpose([2, 0, 1])  # (2, 2, 2) -> (2, 2, 2) but permuted
        np_loss = np_out.sum()
        np_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)
    
    def test_no_transpose(self):
        """Test transpose() with no args (reverse all axes)"""
        X = np.array([[[1.0, 2.0],
                       [3.0, 4.0]],
                      [[5.0, 6.0],
                       [7.0, 8.0]]], dtype=np.float32)  # (2, 2, 2)
        
        # Torch (torch doesn't have transpose() with no args, use permute)
        tx = torch.tensor(X, requires_grad=True)
        torch_out = tx.permute(2, 1, 0)  # reverse all axes
        loss = torch_out.sum()
        loss.backward()
        
        # npgpt
        nx = npgpt.Tensor(X)
        np_out = nx.transpose()  # reverse all axes
        np_loss = np_out.sum()
        np_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)


class TestCombined:
    """Test combinations of operations"""
    
    def test_slice_then_reshape(self):
        """Test x[:, :2].reshape(-1)"""
        X = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], dtype=np.float32)
        
        # Torch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = tx[:, :2].reshape(-1)  # (2,3) -> (2,2) -> (4,)
        loss = torch_out.sum()
        loss.backward()
        
        # npgpt
        nx = npgpt.Tensor(X)
        np_out = nx[:, :2].reshape(4)  # (2,3) -> (2,2) -> (4,)
        np_loss = np_out.sum()
        np_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)
    
    def test_transpose_then_slice(self):
        """Test x.T[0]"""
        X = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], dtype=np.float32)
        
        # Torch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = tx.T[0]  # (2,3) -> (3,2) -> (2,)
        loss = torch_out.sum()
        loss.backward()
        
        # npgpt
        nx = npgpt.Tensor(X)
        np_out = nx.T[0]  # (2,3) -> (3,2) -> (2,)
        np_loss = np_out.sum()
        np_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)
