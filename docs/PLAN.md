# npgpt Autograd + GPT Plan

Goal: Rebuild the minimum set of Tensor primitives, autograd, and composite NN ops in NumPy to train/infer a GPT-like model, while keeping a 1:1 API with PyTorch for easy unit testing against torch.

Guiding principles:
- Match PyTorch tensor APIs where feasible (names, shapes, broadcasting semantics, dtype behavior).
- Every primitive: implement forward in NumPy, write a custom `_backward` with correct broadcasting/reduction, and unit test against PyTorch forward/backward.
- Prefer simple numerics first; optimize/stabilize as needed later (e.g., fused implementations, eps handling).
- Keep ops composable so larger modules (LN, Attention, MLP) can be written either as composite ops or as custom primitives when beneficial.

---

## 0) Scaffolding & Testing
- Package layout: `npgpt/` with `Tensor` core and utilities.
- Experiments and tests under `experiments/` and `tests/`.
- PyTest tests that compare against torch: forward values and gradients.
- Utilities: `sum_to_shape(grad, target_shape)` to reduce broadcasted grads.

Deliverables:
- `npgpt/tensor.py` base class with graph, topo `backward()`.
- `tests/01_basic.py` sanity tests (done).

---

## 1) Tensor Basics
- Dtype/shape helpers: `.shape()`, `.dtype()`.
- Grad management: `.zero_grad()`.
- Item access: `__getitem__`, `__setitem__` (no autograd yet for slicing; add later).
- Creation helpers: `zeros`, `ones`, `randn` wrappers (optional early).

Tests:
- Construction from list/ndarray, correct dtypes.

---

## 2) Elementwise Arithmetic Primitives
Implement as custom primitives with broadcasting support.
- Negation: `__neg__`
- Add/Sub: `__add__`, `__radd__`, `__sub__`, `__rsub__`
- Mul: `__mul__`, `__rmul__`
- Div: `__truediv__`, `__rtruediv__`
- Pow: `__pow__` (scalar and elementwise)

Backward rules (with broadcast reduction via `sum_to_shape`):
- z = x + y -> dx = g, dy = g
- z = x - y -> dx = g, dy = -g
- z = x * y -> dx = g * y, dy = g * x
- z = x / y -> dx = g / y, dy = -g * x / (y**2)
- z = x ** p (scalar p) -> dx = g * p * x**(p-1)

Tests:
- Forward equality to torch
- Backward through `sum()` to scalar and random upstream grads
- Broadcasting cases (e.g., (B,T,C) +- (1,C))

---

## 3) Reductions
- `sum(axis=None, keepdims=False)`
- `mean(axis=None, keepdims=False)`
- `max`/`argmax` (forward first; backward for max via mask)

Backward:
- sum: distribute upstream `g` uniformly to reduced axes
- mean: like sum, scaled by 1/N over reduced dims

Tests:
- Torch parity across axis/keepdims combinations

---

## 4) Linear Algebra
- `matmul` / `@` (2D first, then batched matmul)
- `transpose`/`permute` (2D T first, general permute later)
- `reshape`, `view`-like (no-copy views semantics are hard; start with copies and correct grads)

Backward:
- matmul: dx = g @ W.T, dW = X.T @ g (batched: sum over batch)
- transpose/permute: grad permutes back
- reshape/view: grad reshapes back

Tests:
- Linear layer forward/backward parity with torch
- Batched matmul shapes

---

## 5) Nonlinearities
- `relu`, `gelu` (exact and/or tanh approx)
- `tanh`, `sigmoid`

Backward:
- relu: mask > 0
- gelu: implement derivative for chosen variant
- tanh/sigmoid: standard derivatives

Tests:
- Forward/backward parity on random inputs

---

## 6) Normalization
- `layernorm(x, weight, bias, eps)` as custom primitive (Norm over last dim)
- `rmsnorm` (optional)

Backward:
- Save `mu, var, inv_std, x_hat`; implement param and input grads; use `sum_to_shape` for broadcast params

Tests:
- Parity with torch.nn.LayerNorm for forward/backward

---

## 7) Attention Components
- Projections: linear layers (X @ W + b)
- Scaled dot-product attention primitive: `attn(Q,K,V, mask=None)` returning output and optionally attention weights
- Causal mask handling
- Softmax primitive (with numerical stability)

Backward:
- Softmax: use saved probs; Jacobian-vector product implementation
- Attention: grads through scores, mask (as no-op in backward), and V

Tests:
- Compare against small torch attention forward/backward
- Check masking correctness

---

## 8) Transformer Blocks
- Block = LN -> MHA -> residual -> LN -> MLP -> residual
- MLP = Linear -> GELU -> Linear

Tests:
- Tiny config parity vs torch for few steps (loss, grads)

---

## 9) GPT Model
- Embedding, positional embedding, stack of blocks, final layer norm, LM head
- Weight tying wte <-> lm_head

Tests:
- Forward shape checks
- Cross-entropy loss parity (implement `cross_entropy` primitive)
- Tiny dataset training smoke test

---

## 10) Utilities & Training
- Optimizers: SGD/AdamW minimal
- Dtype control (float32 baseline)
- Device stubs (CPU only)
- Seed control

---

## Testing Strategy
- Each primitive has forward/backward tests vs torch
- Randomized shapes with broadcasting
- Numeric grad checks (finite differences) for new ops initially

---

## File Map (proposed)
- `npgpt/tensor.py` — core Tensor + primitive ops
- `npgpt/nn/functions.py` — custom primitives (layernorm, softmax, attention)
- `npgpt/nn/modules.py` — modules composed from primitives (Linear, LayerNorm, MLP, Attention, Block, GPT)
- `npgpt/utils.py` — sum_to_shape, helpers
- `tests/` — unit tests mirroring PyTorch behavior
- `experiments/` — ad-hoc scripts

---

## Milestone Checklist
- [ ] Elementwise: +, -, *, /, pow, neg
- [ ] Reductions: sum/mean(axis, keepdims)
- [ ] Shape ops: reshape, transpose
- [ ] Matmul (2D and batched)
- [ ] Nonlinearities: relu, tanh, sigmoid, gelu
- [ ] LayerNorm
- [ ] Softmax
- [ ] Attention
- [ ] Linear, MLP
- [ ] Transformer Block
- [ ] GPT
