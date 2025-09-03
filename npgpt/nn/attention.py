import numpy as np
import npgpt
from .module import Module

class MultiHeadAttention(Module):
    def __init__(self, embed_dim, num_heads, causal=True):
        super().__init__()
        
        self.D = embed_dim
        self.N = num_heads
        assert self.D % self.N == 0, "embed_dim must be divisible by num_heads"
        self.H = self.D // self.N  # head dimension
        
        self.causal = causal
        
        # QKV Projection Weights - register as parameters
        self.add_parameter('W_q', npgpt.Tensor(np.random.randn(self.D, self.D) * 0.01))
        self.add_parameter('W_k', npgpt.Tensor(np.random.randn(self.D, self.D) * 0.01))
        self.add_parameter('W_v', npgpt.Tensor(np.random.randn(self.D, self.D) * 0.01))
        
        # Output projection Weights
        self.add_parameter('W_o', npgpt.Tensor(np.random.randn(self.D, self.D) * 0.01))
        
    def forward(self, t: npgpt.Tensor):
        # T: (batch_size, seq_len, embed_dim)
        assert t.shape()[-1] == self.D
        
        B, S, _ = t.shape()
        
        # Get parameters
        W_q = self._parameters['W_q']
        W_k = self._parameters['W_k'] 
        W_v = self._parameters['W_v']
        W_o = self._parameters['W_o']
        
        # attention == softmax(q@k.t)
        q = t @ W_q  # (B, S, D)
        k = t @ W_k  # (B, S, D)
        v = t @ W_v  # (B, S, D)
        
        # reshape to multiple heads
        q = q.reshape(B, S, self.N, self.H).transpose((0, 2, 1, 3))  # (B, N, S, H)
        k = k.reshape(B, S, self.N, self.H).transpose((0, 2, 1, 3))  # (B, N, S, H)
        v = v.reshape(B, S, self.N, self.H).transpose((0, 2, 1, 3))  # (B, N, S, H)
        
        # Attention scores: Q @ K^T
        att_scores = (q @ k.transpose((0, 1, 3, 2))) / np.sqrt(self.H)  # (B, N, S, S)
        
        if self.causal:
            # Create causal mask that preserves gradients
            S = att_scores.shape()[-1]
            mask = np.tril(np.ones((S, S), dtype=bool))
            # Apply mask by adding large negative value to masked positions
            mask_value = -1e9
            mask_tensor = npgpt.Tensor(np.where(mask, 0.0, mask_value))
            att_scores = att_scores + mask_tensor
        
        att_weights = npgpt.softmax(att_scores, axis=-1)  # (B, N, S, S)
        att_output = att_weights @ v  # (B, N, S, H)
        
        # Reshape back to original format
        att_output = att_output.transpose((0, 2, 1, 3)).reshape(B, S, self.D)  # (B, S, D)
        out = att_output @ W_o  # (B, S, D)
        
        return out
            
        
        
        
        
        