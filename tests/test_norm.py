import numpy as np
import torch
import torch.nn as nn
import npgpt
from npgpt.utils import np_allclose

"""
Tests for normalization operations (LayerNorm) comparing against PyTorch.

Run: pytest tests/test_norm.py -sv
"""


class TestLayerNorm:
    """Tests for LayerNorm normalization"""
    
    def test_layernorm_forward_backward(self):
        """Test LayerNorm forward and backward pass with standard inputs"""
        batch_size, seq_len, embed_dim = 2, 4, 8
        
        # Input data
        X = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
        # Now using PyTorch-compatible 1D shapes
        gamma_data = np.ones(embed_dim, dtype=np.float32)
        beta_data = np.zeros(embed_dim, dtype=np.float32)
        
        # PyTorch
        tx = torch.tensor(X, requires_grad=True)
        torch_ln = nn.LayerNorm(embed_dim)
        torch_ln.weight.data = torch.tensor(gamma_data)
        torch_ln.bias.data = torch.tensor(beta_data)
        torch_out = torch_ln(tx)
        loss = torch_out.sum()
        loss.backward()
        
        # npgpt
        nx = npgpt.Tensor(X)
        ngamma = npgpt.Tensor(gamma_data)
        nbeta = npgpt.Tensor(beta_data)
        np_out = npgpt.layernorm(nx, ngamma, nbeta)
        np_loss = np_out.sum()
        np_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data, rtol=1e-4, atol=1e-6)
        assert np_allclose(tx.grad.numpy(), nx.grad, rtol=1e-3, atol=1e-6)
        assert np_allclose(torch_ln.weight.grad.numpy(), ngamma.grad, rtol=1e-4, atol=1e-6)
        assert np_allclose(torch_ln.bias.grad.numpy(), nbeta.grad, rtol=1e-4, atol=1e-6)
    
    def test_layernorm_chain_operations(self):
        """Test LayerNorm in a chain of operations"""
        batch_size, seq_len, embed_dim = 2, 3, 4
        
        X = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
        gamma_data = np.random.randn(embed_dim).astype(np.float32)
        beta_data = np.random.randn(embed_dim).astype(np.float32)
        W = np.random.randn(embed_dim, embed_dim).astype(np.float32)
        
        # PyTorch: X -> LayerNorm -> matmul -> sum
        tx = torch.tensor(X, requires_grad=True)
        tw = torch.tensor(W, requires_grad=True)
        torch_ln = nn.LayerNorm(embed_dim)
        torch_ln.weight.data = torch.tensor(gamma_data)
        torch_ln.bias.data = torch.tensor(beta_data)
        torch_norm = torch_ln(tx)
        torch_out = torch_norm @ tw
        loss = torch_out.sum()
        loss.backward()
        
        # npgpt: X -> LayerNorm -> matmul -> sum
        nx = npgpt.Tensor(X)
        nw = npgpt.Tensor(W)
        ngamma = npgpt.Tensor(gamma_data)
        nbeta = npgpt.Tensor(beta_data)
        np_norm = npgpt.layernorm(nx, ngamma, nbeta)
        np_out = np_norm @ nw
        np_loss = np_out.sum()
        np_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data, rtol=1e-4, atol=1e-6)
        assert np_allclose(tx.grad.numpy(), nx.grad, rtol=1e-3, atol=1e-6)
        assert np_allclose(tw.grad.numpy(), nw.grad, rtol=1e-4, atol=1e-6)
        assert np_allclose(torch_ln.weight.grad.numpy(), ngamma.grad, rtol=1e-4, atol=1e-6)
        assert np_allclose(torch_ln.bias.grad.numpy(), nbeta.grad, rtol=1e-4, atol=1e-6)

    def test_layernorm_simple_manual_backward(self):
        """Manually verify LayerNorm gradients for simple case"""
        # Simple case: single batch, single sequence position
        embed_dim = 3
        X = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # (1, 3)
        gamma_data = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # (3,)
        beta_data = np.array([0.0, 0.0, 0.0], dtype=np.float32)   # (3,)
        
        nx = npgpt.Tensor(X)
        ngamma = npgpt.Tensor(gamma_data)
        nbeta = npgpt.Tensor(beta_data)
        
        # Forward pass
        np_out = npgpt.layernorm(nx, ngamma, nbeta)
        loss = np_out.sum()
        
        # Backward pass
        loss.backward()
        
        # Manual verification using PyTorch
        tx = torch.tensor(X, requires_grad=True)
        torch_ln = nn.LayerNorm(embed_dim)
        torch_ln.weight.data = torch.tensor(gamma_data)
        torch_ln.bias.data = torch.tensor(beta_data)
        torch_out = torch_ln(tx)
        torch_loss = torch_out.sum()
        torch_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data, rtol=1e-4, atol=1e-6)
        assert np_allclose(tx.grad.numpy(), nx.grad, rtol=1e-3, atol=1e-6)
        assert np_allclose(torch_ln.weight.grad.numpy(), ngamma.grad, rtol=1e-4, atol=1e-6)
        assert np_allclose(torch_ln.bias.grad.numpy(), nbeta.grad, rtol=1e-4, atol=1e-6)

    def test_layernorm_batch_sequence_embedding(self):
        """Test LayerNorm with realistic transformer dimensions"""
        np.random.seed(42)
        batch_size, seq_len, embed_dim = 3, 5, 16
        
        X = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
        gamma_data = np.random.randn(embed_dim).astype(np.float32) * 0.1 + 1.0  # Around 1.0
        beta_data = np.random.randn(embed_dim).astype(np.float32) * 0.1  # Around 0.0
        
        # PyTorch
        tx = torch.tensor(X, requires_grad=True)
        torch_ln = nn.LayerNorm(embed_dim)
        torch_ln.weight.data = torch.tensor(gamma_data)
        torch_ln.bias.data = torch.tensor(beta_data)
        torch_out = torch_ln(tx)
        torch_loss = torch_out.mean()  # Use mean to keep gradients reasonable
        torch_loss.backward()
        
        # npgpt
        nx = npgpt.Tensor(X)
        ngamma = npgpt.Tensor(gamma_data)
        nbeta = npgpt.Tensor(beta_data)
        np_out = npgpt.layernorm(nx, ngamma, nbeta)
        np_loss = np_out.mean()
        np_loss.backward()
        
        assert np_allclose(torch_out.detach().numpy(), np_out.data, rtol=1e-4, atol=1e-6)
        assert np_allclose(tx.grad.numpy(), nx.grad, rtol=1e-3, atol=1e-6)
        assert np_allclose(torch_ln.weight.grad.numpy(), ngamma.grad, rtol=1e-4, atol=1e-6)
        assert np_allclose(torch_ln.bias.grad.numpy(), nbeta.grad, rtol=1e-4, atol=1e-6)

    def test_layernorm_zero_variance_handling(self):
        """Test LayerNorm handles near-zero variance gracefully"""
        embed_dim = 4
        # Create input with very small variance
        X = np.array([[1.000001, 1.000002, 1.000001, 1.000002]], dtype=np.float32)
        gamma_data = np.ones(embed_dim, dtype=np.float32)
        beta_data = np.zeros(embed_dim, dtype=np.float32)
        
        # PyTorch
        tx = torch.tensor(X, requires_grad=True)
        torch_ln = nn.LayerNorm(embed_dim, eps=1e-5)
        torch_ln.weight.data = torch.tensor(gamma_data)
        torch_ln.bias.data = torch.tensor(beta_data)
        torch_out = torch_ln(tx)
        torch_loss = torch_out.sum()
        torch_loss.backward()
        
        # npgpt
        nx = npgpt.Tensor(X)
        ngamma = npgpt.Tensor(gamma_data)
        nbeta = npgpt.Tensor(beta_data)
        np_out = npgpt.layernorm(nx, ngamma, nbeta, eps=1e-5)
        np_loss = np_out.sum()
        np_loss.backward()
        
        # Should not have NaN or Inf values
        assert np.all(np.isfinite(np_out.data))
        assert np.all(np.isfinite(nx.grad))
        assert np.all(np.isfinite(ngamma.grad))
        assert np.all(np.isfinite(nbeta.grad))
        
        # Should match PyTorch
        assert np_allclose(torch_out.detach().numpy(), np_out.data, rtol=1e-4, atol=1e-6)
        # For near-zero variance, PyTorch may return zero gradients while we compute very small ones
        # Use very relaxed tolerance for this edge case
        torch_grad = tx.grad.numpy()
        if np.allclose(torch_grad, 0.0, atol=1e-8):
            # If PyTorch returns effectively zero gradients, our gradients should be very small
            assert np.all(np.abs(nx.grad) < 1e-5), f"Expected small gradients, got {nx.grad}"
        else:
            assert np_allclose(torch_grad, nx.grad, rtol=1e-3, atol=1e-6)
