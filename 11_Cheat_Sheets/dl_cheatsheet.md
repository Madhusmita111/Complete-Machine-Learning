#  Deep Learning Cheat Sheet

> All critical DL concepts, architectures, and formulas in one place.

---

##  Neural Network Basics

```
Input Layer → Hidden Layers → Output Layer

Each neuron:   z = Wx + b
               a = activation(z)

Full forward pass (L layers):
  a⁰ = x
  zˡ = Wˡaˡ⁻¹ + bˡ
  aˡ = g(zˡ)   ← g is activation function
  ŷ = aᴸ
```

---

##  Activation Functions

| Name | Formula | Derivative | Range | Use |
|---|---|---|---|---|
| Sigmoid | 1/(1+e⁻ˣ) | σ(1-σ) | (0,1) | Binary output |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | 1-tanh² | (-1,1) | Hidden (older) |
| ReLU | max(0,x) | 0 or 1 | [0,∞) | **Default hidden** |
| Leaky ReLU | max(0.01x, x) | 0.01 or 1 | (-∞,∞) | Dying ReLU fix |
| ELU | x if x≥0, α(eˣ-1) else | — | (-α,∞) | Smoother ReLU |
| Softmax | eˣⁱ/Σeˣʲ | — | (0,1), Σ=1 | Multiclass output |
| GELU | x·Φ(x) | — | — | Transformers |
| Swish | x·sigmoid(x) | — | — | EfficientNet |

**Dying ReLU:** Neurons stuck at 0 → gradient = 0 → no learning. Fix: Leaky ReLU.

---

##  Backpropagation

```
Forward:  Compute all activations & loss L

Backward (chain rule):
  dL/dWˡ = dL/daˡ · daˡ/dzˡ · dzˡ/dWˡ
          = δˡ · (aˡ⁻¹)ᵀ

  where δˡ = (dL/daˡ) ⊙ g'(zˡ)    ← element-wise

Update:
  Wˡ ← Wˡ - α · dL/dWˡ
  bˡ ← bˡ - α · dL/dbˡ
```

---

##  Optimizers

| Optimizer | Update Rule | Best When |
|---|---|---|
| SGD | θ ← θ - α∇L | Simple, stable |
| SGD + Momentum | v ← γv + α∇L; θ ← θ-v | Faster than SGD |
| RMSProp | Divide by running avg of squared grads | Non-stationary problems |
| Adam | 1st + 2nd moment with bias correction | **Default choice** |
| AdamW | Adam + weight decay fix | Transformers, modern models |
| Adagrad | Per-param lr, decreasing | Sparse gradients, NLP |

**Adam defaults:** α=0.001, β₁=0.9, β₂=0.999, ε=1e-8

---

##  Learning Rate Strategies

| Strategy | Behavior | When |
|---|---|---|
| Fixed | Constant | Simple problems |
| Step Decay | Reduce by factor every N epochs | Common default |
| Cosine Annealing | Smooth decrease following cosine | General use |
| Warmup + Decay | Increase then decrease | Transformers |
| Cyclical LR | Cycle between min and max | Finding optimal LR |
| One Cycle Policy | Fast.ai's strategy | Often fastest convergence |

---

##  Regularization Techniques

| Technique | What it does | Where |
|---|---|---|
| L1/L2 weight decay | Penalizes large weights | Loss function |
| Dropout | Randomly zeros neurons | Hidden layers |
| Batch Normalization | Normalizes layer activations | After linear, before activation |
| Layer Normalization | Like BN but over features | Transformers |
| Early Stopping | Stop at lowest val loss | Training loop |
| Data Augmentation | Increases data diversity | Input pipeline |
| Label Smoothing | Softens targets (not 0/1) | Loss function |
| Gradient Clipping | Caps gradient magnitude | Training (RNNs, Transformers) |

---

##  CNN Architecture

```
Input Image → [Conv → BN → ReLU → Pool] × N → Flatten → FC → Output

Key operations:
  Convolution:   Feature map = Input ★ Kernel + bias
  Pooling:       Reduces spatial dimensions
  Stride:        Step size of kernel
  Padding:       'same' keeps shape, 'valid' reduces

Output size formula:
  W_out = (W_in - F + 2P) / S + 1
  W = width, F = filter size, P = padding, S = stride
```

**Classic Architectures:**
| Model | Key Innovation | Year |
|---|---|---|
| LeNet-5 | First practical CNN | 1998 |
| AlexNet | Deep CNN + ReLU + Dropout | 2012 |
| VGGNet | Deep with 3×3 convs only | 2014 |
| GoogLeNet/Inception | Parallel convolutions | 2014 |
| ResNet | Skip connections | 2015 |
| DenseNet | Dense connections | 2016 |
| EfficientNet | Compound scaling | 2019 |
| ViT | Vision Transformer | 2020 |

---

##  RNN Family

```
Vanilla RNN:
  hₜ = tanh(Wₕhₜ₋₁ + Wₓxₜ + b)
  yₜ = Wᵧhₜ

Problem: Vanishing/exploding gradients over long sequences

LSTM:
  fₜ = σ(Wf[hₜ₋₁, xₜ] + bf)     ← Forget gate
  iₜ = σ(Wᵢ[hₜ₋₁, xₜ] + bᵢ)     ← Input gate
  C̃ₜ = tanh(Wc[hₜ₋₁, xₜ] + bc)   ← Candidate
  Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙C̃ₜ           ← Cell state update
  oₜ = σ(Wo[hₜ₋₁, xₜ] + bo)     ← Output gate
  hₜ = oₜ ⊙ tanh(Cₜ)              ← Hidden state

GRU (simplified LSTM):
  zₜ = σ(Wz[hₜ₋₁, xₜ])           ← Update gate
  rₜ = σ(Wr[hₜ₋₁, xₜ])           ← Reset gate
  h̃ₜ = tanh(W[rₜ⊙hₜ₋₁, xₜ])      ← Candidate
  hₜ = (1-zₜ)⊙hₜ₋₁ + zₜ⊙h̃ₜ      ← Final hidden
```

---

##  Transformer Architecture

```
Encoder block:
  x → MultiHead Self-Attention → Add & Norm → FFN → Add & Norm

Decoder block:
  y → Masked MultiHead Attention → Add & Norm
    → Cross-Attention (with encoder output) → Add & Norm
    → FFN → Add & Norm

Attention:
  Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V

Positional Encoding:
  PE(pos, 2i)   = sin(pos / 10000^(2i/d))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**BERT:** Encoder only, masked LM + next sentence prediction
**GPT:** Decoder only, causal (left-to-right) language modeling
**T5:** Encoder-Decoder, text-to-text framework

---

##  Common Loss Functions

| Task | Loss | Notes |
|---|---|---|
| Binary classification | Binary cross-entropy | Use sigmoid output |
| Multiclass | Categorical cross-entropy | Use softmax output |
| Regression | MSE / MAE / Huber | MAE more robust |
| Image generation | Perceptual loss, adversarial loss | GANs |
| Embedding | Triplet loss, contrastive | Similarity learning |
| Object detection | Focal loss | Handles class imbalance |

---

##  Training Tips

```
Checklist for training deep models:
  ✅ Normalize inputs (zero mean, unit variance)
  ✅ Initialize weights properly (He for ReLU)
  ✅ Use Adam optimizer (default)
  ✅ Start with learning rate ~1e-3
  ✅ Use learning rate scheduler
  ✅ Monitor training AND validation loss
  ✅ Use Batch Normalization
  ✅ Add Dropout in FC layers
  ✅ Clip gradients if using RNNs
  ✅ Use early stopping
  ✅ Augment data if small dataset
```

---

## 🔧 Weight Initialization

| Method | Formula | For |
|---|---|---|
| Zero | w = 0 | ❌ Never use (symmetry breaking) |
| Random Normal | w ~ N(0, 0.01) | Simple, but may vanish |
| Xavier/Glorot | w ~ U[-√(6/(nᵢ+nₒ)), ...] | Sigmoid, Tanh |
| He/Kaiming | w ~ N(0, √(2/nᵢₙ)) | **ReLU** (recommended) |
| Orthogonal | Orthogonal matrix | RNNs |

---

##  Batch Size Effects

| Batch Size | Effect |
|---|---|
| Very small (1-4) | Noisy gradients; acts as regularizer; slow convergence |
| Small (16-64) | Better generalization; more noise |
| Large (256-1024) | Stable gradients; faster per epoch; may overfit |
| Very large | Sharp minima; worse generalization; needs lr scaling |

**Rule of thumb:** Scale lr proportionally when changing batch size. If batch ×2, lr ×2.

---

*Part of Complete ML Notes Repository*
