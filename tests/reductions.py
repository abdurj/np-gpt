import numpy as np
import torch
import npgpt
import pytest

"""
Tests for reduction operations (sum, mean) comparing against PyTorch.

Run: pytest tests/reductions.py -sv
"""

def np_allclose(a, b, rtol=1e-5, atol=1e-7):
    return np.allclose(a, b, rtol=rtol, atol=atol)

# ----- Test Classes (Alternative Organization) -------- #

class TestSum:
    """All sum-related tests grouped in a class"""

    def test_all_elements(self):
        """Test sum() with no axis (sum all elements to scalar)"""
        X = np.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]], dtype=np.float32)

        # Torch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = tx.sum()
        torch_out.backward()

        # npgpt
        nx = npgpt.Tensor(X)
        np_out = nx.sum()
        np_out.backward()

        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)


    def test_axis_0(self):
        """Test sum(axis=0) - sum over rows"""
        X = np.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]], dtype=np.float32)  # (2,3)

        # Torch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = tx.sum(dim=0)  # -> (3,)
        loss = torch_out.sum()  # reduce to scalar for backward
        loss.backward()

        # npgpt
        nx = npgpt.Tensor(X)
        np_out = nx.sum(axis=0)  # -> (3,)
        np_loss = np_out.sum()
        np_loss.backward()

        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)


    def test_axis_1(self):
        """Test sum(axis=1) - sum over columns"""
        X = np.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]], dtype=np.float32)  # (2,3)

        # Torch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = tx.sum(dim=1)  # -> (2,)
        loss = torch_out.sum()
        loss.backward()

        # npgpt
        nx = npgpt.Tensor(X)
        np_out = nx.sum(axis=1)  # -> (2,)
        np_loss = np_out.sum()
        np_loss.backward()

        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)


    def test_keepdims(self):
        """Test sum(axis=0, keepdims=True)"""
        X = np.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]], dtype=np.float32)  # (2,3)

        # Torch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = tx.sum(dim=0, keepdim=True)  # -> (1,3)
        loss = torch_out.sum()
        loss.backward()

        # npgpt
        nx = npgpt.Tensor(X)
        np_out = nx.sum(axis=0, keepdims=True)  # -> (1,3)
        np_loss = np_out.sum()
        np_loss.backward()

        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)
        
    def test_multiple_axes(self):
        """Test sum over multiple axes"""
        X = np.array([[[1.0, 2.0],
                    [3.0, 4.0]],
                    [[5.0, 6.0],
                    [7.0, 8.0]]], dtype=np.float32)  # (2,2,2)

        # Torch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = tx.sum(dim=(0, 2))  # -> (2,)
        loss = torch_out.sum()
        loss.backward()

        # npgpt
        nx = npgpt.Tensor(X)
        np_out = nx.sum(axis=(0, 2))  # -> (2,)
        np_loss = np_out.sum()
        np_loss.backward()

        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)


    def test_negative_axis(self):
        """Test sum with negative axis"""
        X = np.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]], dtype=np.float32)  # (2,3)

        # Torch
        tx = torch.tensor(X, requires_grad=True)
        torch_out = tx.sum(dim=-1)  # same as dim=1, -> (2,)
        loss = torch_out.sum()
        loss.backward()

        # npgpt
        nx = npgpt.Tensor(X)
        np_out = nx.sum(axis=-1)  # same as axis=1, -> (2,)
        np_loss = np_out.sum()
        np_loss.backward()

        assert np_allclose(torch_out.detach().numpy(), np_out.data)
        assert np_allclose(tx.grad.numpy(), nx.grad)



# class TestMean:
#     """All mean-related tests grouped in a class"""
        
#     def test_all_elements(self):
#         """Test mean() with no axis"""
#         X = np.array([[1.0, 2.0, 3.0],
#                     [4.0, 5.0, 6.0]], dtype=np.float32)

#         # Torch
#         tx = torch.tensor(X, requires_grad=True)
#         torch_out = tx.mean()
#         torch_out.backward()

#         # npgpt
#         nx = npgpt.Tensor(X)
#         np_out = nx.mean()
#         np_out.backward()

#         assert np_allclose(torch_out.detach().numpy(), np_out.data)
#         assert np_allclose(tx.grad.numpy(), nx.grad)


#     def test_axis_0(self):
#         """Test mean(axis=0)"""
#         X = np.array([[1.0, 2.0, 3.0],
#                     [4.0, 5.0, 6.0]], dtype=np.float32)  # (2,3)

#         # Torch
#         tx = torch.tensor(X, requires_grad=True)
#         torch_out = tx.mean(dim=0)  # -> (3,)
#         loss = torch_out.sum()
#         loss.backward()

#         # npgpt
#         nx = npgpt.Tensor(X)
#         np_out = nx.mean(axis=0)  # -> (3,)
#         np_loss = np_out.sum()
#         np_loss.backward()

#         assert np_allclose(torch_out.detach().numpy(), np_out.data)
#         assert np_allclose(tx.grad.numpy(), nx.grad)


#     def test_axis_1_keepdims(self):
#         """Test mean(axis=1, keepdims=True)"""
#         X = np.array([[1.0, 2.0, 3.0],
#                     [4.0, 5.0, 6.0]], dtype=np.float32)  # (2,3)

#         # Torch
#         tx = torch.tensor(X, requires_grad=True)
#         torch_out = tx.mean(dim=1, keepdim=True)  # -> (2,1)
#         loss = torch_out.sum()
#         loss.backward()

#         # npgpt
#         nx = npgpt.Tensor(X)
#         np_out = nx.mean(axis=1, keepdims=True)  # -> (2,1)
#         np_loss = np_out.sum()
#         np_loss.backward()

#         assert np_allclose(torch_out.detach().numpy(), np_out.data)
#         assert np_allclose(tx.grad.numpy(), nx.grad)
