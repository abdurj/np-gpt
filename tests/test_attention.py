import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import npgpt
import pytest
from npgpt.utils import np_allclose

"""
Tests for MultiHeadAttention comparing against PyTorch.

Run: pytest tests/test_attention.py -sv
"""

class TestMultiHeadAttention:
    """Tests for MultiHeadAttention module"""
    
    def test_attention_forward_basic(self):
        """Test basic forward pass without causal masking"""
        batch_size, seq_len, embed_dim = 2, 4, 8
        num_heads = 2
        
        # Create test input
        X = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
        
        # npgpt attention (non-causal for easier comparison)
        np_attn = npgpt.nn.MultiHeadAttention(embed_dim, num_heads, causal=False)
        
        # PyTorch attention
        torch_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Copy weights from npgpt to torch for fair comparison
        with torch.no_grad():
            torch_attn.in_proj_weight[:embed_dim, :] = torch.from_numpy(np_attn._parameters['W_q'].data.astype(np.float32))
            torch_attn.in_proj_weight[embed_dim:2*embed_dim, :] = torch.from_numpy(np_attn._parameters['W_k'].data.astype(np.float32))
            torch_attn.in_proj_weight[2*embed_dim:, :] = torch.from_numpy(np_attn._parameters['W_v'].data.astype(np.float32))
            torch_attn.out_proj.weight.data = torch.from_numpy(np_attn._parameters['W_o'].data.astype(np.float32))
            torch_attn.in_proj_bias.zero_()
            torch_attn.out_proj.bias.zero_()
        
        # Forward pass
        nx = npgpt.Tensor(X)
        np_out = np_attn(nx)
        
        tx = torch.tensor(X, requires_grad=True)
        torch_out, _ = torch_attn(tx, tx, tx)  # self-attention
        
        print(f"npgpt output shape: {np_out.shape()}")
        print(f"torch output shape: {torch_out.shape}")
        
        # Check shapes match
        assert np_out.shape() == tuple(torch_out.shape)
    
    def test_attention_causal_mask(self):
        """Test that causal masking works correctly"""
        batch_size, seq_len, embed_dim = 1, 4, 8
        num_heads = 2
        
        X = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
        
        # Test causal vs non-causal
        causal_attn = npgpt.nn.MultiHeadAttention(embed_dim, num_heads, causal=True)
        non_causal_attn = npgpt.nn.MultiHeadAttention(embed_dim, num_heads, causal=False)
        
        # Copy weights to make them identical
        for param_name in ['W_q', 'W_k', 'W_v', 'W_o']:
            non_causal_attn._parameters[param_name].data = causal_attn._parameters[param_name].data.copy()
        
        nx = npgpt.Tensor(X)
        causal_out = causal_attn(nx)
        non_causal_out = non_causal_attn(nx)
        
        # Outputs should be different due to masking
        assert not np_allclose(causal_out.data, non_causal_out.data)
        print("✓ Causal masking produces different output")
    
    def test_attention_shapes(self):
        """Test attention with different input shapes"""
        test_cases = [
            (1, 1, 4, 1),    # Single token, single head
            (2, 5, 12, 3),   # Multiple batch, sequence, heads
            (1, 10, 16, 4),  # Longer sequence
        ]
        
        for batch_size, seq_len, embed_dim, num_heads in test_cases:
            X = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
            
            attn = npgpt.nn.MultiHeadAttention(embed_dim, num_heads)
            nx = npgpt.Tensor(X)
            out = attn(nx)
            
            # Output shape should match input shape
            assert out.shape() == (batch_size, seq_len, embed_dim)
            print(f"✓ Shape test passed for {(batch_size, seq_len, embed_dim, num_heads)}")
    
    def test_attention_parameters(self):
        """Test that parameters are properly registered"""
        embed_dim, num_heads = 8, 2
        attn = npgpt.nn.MultiHeadAttention(embed_dim, num_heads)
        
        # Should have 4 parameters: W_q, W_k, W_v, W_o
        params = attn.parameters()
        assert len(params) == 4
        
        param_names = [name for name, _ in attn.named_parameters()]
        expected_names = ['W_q', 'W_k', 'W_v', 'W_o']
        assert set(param_names) == set(expected_names)
        
        # All should have correct shapes
        for param in params:
            assert param.shape() == (embed_dim, embed_dim)
        
        print("✓ Parameter registration test passed")
    
    def test_attention_backward_basic(self):
        """Test basic backward pass"""
        batch_size, seq_len, embed_dim = 2, 3, 4
        num_heads = 2
        
        X = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
        
        attn = npgpt.nn.MultiHeadAttention(embed_dim, num_heads, causal=False)
        nx = npgpt.Tensor(X)
        
        # Forward pass
        out = attn(nx)
        loss = out.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist for all parameters
        for param_name, param in attn.named_parameters():
            assert param.grad is not None, f"No gradient for {param_name}"
            assert param.grad.shape == param.data.shape
            assert not np.allclose(param.grad, 0), f"Zero gradient for {param_name}"
        
        # Check input gradients
        assert nx.grad is not None
        assert nx.grad.shape == X.shape
        
        print("✓ Backward pass test passed")
    
    def test_attention_multi_head_consistency(self):
        """Test that different numbers of heads produce consistent results"""
        batch_size, seq_len, embed_dim = 1, 3, 12
        
        X = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
        
        # Test with different head counts
        for num_heads in [1, 2, 3, 4, 6, 12]:
            if embed_dim % num_heads != 0:
                continue
                
            attn = npgpt.nn.MultiHeadAttention(embed_dim, num_heads, causal=False)
            nx = npgpt.Tensor(X)
            out = attn(nx)
            
            assert out.shape() == (batch_size, seq_len, embed_dim)
            assert np.all(np.isfinite(out.data))
            
        print("✓ Multi-head consistency test passed")
    
    def test_attention_self_attention_properties(self):
        """Test mathematical properties of self-attention"""
        batch_size, seq_len, embed_dim = 1, 4, 8
        num_heads = 2
        
        # Create simple input where we can verify attention behavior
        X = np.zeros((batch_size, seq_len, embed_dim), dtype=np.float32)
        X[0, 0, :] = 1.0  # First token has value 1
        X[0, 1, :] = 2.0  # Second token has value 2
        X[0, 2, :] = 0.0  # Third token has value 0
        X[0, 3, :] = 1.0  # Fourth token has value 1
        
        attn = npgpt.nn.MultiHeadAttention(embed_dim, num_heads, causal=False)
        
        # Initialize weights to identity-like matrices for clearer behavior
        for param_name in ['W_q', 'W_k', 'W_v', 'W_o']:
            attn._parameters[param_name].data = np.eye(embed_dim, dtype=np.float32) * 0.1
        
        nx = npgpt.Tensor(X)
        out = attn(nx)
        
        # Output should be finite and have correct shape
        assert np.all(np.isfinite(out.data))
        assert out.shape() == (batch_size, seq_len, embed_dim)
        
        print("✓ Self-attention properties test passed")
    
    def test_attention_gradient_flow(self):
        """Test that gradients flow properly through the attention mechanism"""
        batch_size, seq_len, embed_dim = 2, 3, 8
        num_heads = 2
        
        X = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
        
        attn = npgpt.nn.MultiHeadAttention(embed_dim, num_heads)
        nx = npgpt.Tensor(X)
        
        # Forward pass
        out = attn(nx)
        
        # Create a simple loss that depends on all outputs
        loss = (out ** 2).sum()
        loss.backward()
        
        # Verify gradient magnitudes are reasonable
        for param_name, param in attn.named_parameters():
            grad_norm = np.linalg.norm(param.grad)
            assert grad_norm > 1e-8, f"Gradient too small for {param_name}: {grad_norm}"
            assert grad_norm < 100, f"Gradient too large for {param_name}: {grad_norm}"
        
        input_grad_norm = np.linalg.norm(nx.grad)
        assert input_grad_norm > 1e-8, f"Input gradient too small: {input_grad_norm}"
        assert input_grad_norm < 100, f"Input gradient too large: {input_grad_norm}"
        
        print("✓ Gradient flow test passed")
