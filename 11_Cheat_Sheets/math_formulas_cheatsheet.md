# Math Formulas Cheat Sheet for ML

> Every important formula you need — organized by topic.

---

## 1️⃣ Linear Algebra

### Vectors
```
Dot product:         a · b = Σ aᵢbᵢ = |a||b|cos(θ)
Norm (L2):           ||x|| = √(Σ xᵢ²)
Norm (L1):           ||x||₁ = Σ |xᵢ|
Cosine similarity:   cos(θ) = (a · b) / (||a|| ||b||)
```

### Matrices
```
Matrix multiply:     (AB)ᵢⱼ = Σₖ Aᵢₖ Bₖⱼ
Transpose:           (AB)ᵀ = BᵀAᵀ
Inverse:             A⁻¹A = I  (square, non-singular)
Determinant:         det(A) ≠ 0 → invertible
Trace:               tr(A) = Σ Aᵢᵢ = sum of eigenvalues
Frobenius norm:      ||A||_F = √(Σᵢⱼ Aᵢⱼ²)
```

### Eigendecomposition
```
Eigenvalue eq:   Av = λv
Decomposition:   A = QΛQᵀ  (if A is symmetric)
PCA:             Cov(X) = (1/n) XᵀX → eigenvectors = principal components
```

---

## 2️⃣ Calculus & Optimization

### Derivatives
```
Chain rule:         d/dx[f(g(x))] = f'(g(x)) · g'(x)
Product rule:       d/dx[f·g] = f'g + fg'
Common derivatives: d/dx[eˣ] = eˣ
                    d/dx[ln x] = 1/x
                    d/dx[sigmoid(x)] = σ(x)(1 - σ(x))
```

### Gradient Descent
```
Update rule:    θ ← θ - α ∇_θ L(θ)

Variants:
  SGD:          θ ← θ - α ∇_θ L(θ; xᵢ, yᵢ)
  Momentum:     v ← γv + α∇L;  θ ← θ - v
  Adam:         mₜ = β₁mₜ₋₁ + (1-β₁)gₜ         (1st moment)
                vₜ = β₂vₜ₋₁ + (1-β₂)gₜ²         (2nd moment)
                m̂ₜ = mₜ/(1-β₁ᵗ);  v̂ₜ = vₜ/(1-β₂ᵗ)
                θ ← θ - α·m̂ₜ / (√v̂ₜ + ε)
```

### Backpropagation (Chain Rule Applied)
```
∂L/∂w = ∂L/∂ŷ · ∂ŷ/∂z · ∂z/∂w

where z = wx + b  and  ŷ = σ(z)
```

---

## 3️⃣ Probability & Statistics

### Basic Probability
```
P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
P(A ∩ B) = P(A) · P(B|A)
P(A ∩ B) = P(A) · P(B)   if A, B independent

Conditional:    P(A|B) = P(A ∩ B) / P(B)
```

### Bayes' Theorem
```
P(A|B) = P(B|A) · P(A) / P(B)

Posterior = Likelihood × Prior / Evidence

In ML: P(class | features) ∝ P(features | class) × P(class)
```

### Distributions
```
Normal:       f(x) = (1/√(2πσ²)) · exp(-(x-μ)²/2σ²)
              Mean = μ,  Variance = σ²

Bernoulli:    P(X=1) = p,  P(X=0) = 1-p
              Mean = p,  Variance = p(1-p)

Binomial:     P(X=k) = C(n,k) pᵏ(1-p)ⁿ⁻ᵏ
              Mean = np,  Variance = np(1-p)

Poisson:      P(X=k) = (λᵏe⁻λ)/k!
              Mean = λ,  Variance = λ
```

### Key Statistics
```
Mean:         μ = (1/n) Σ xᵢ
Variance:     σ² = (1/n) Σ(xᵢ - μ)²  [population]
              s² = (1/(n-1)) Σ(xᵢ - μ)²  [sample]
Std Dev:      σ = √(Variance)
Covariance:   Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)]
Correlation:  ρ = Cov(X,Y) / (σₓ · σᵧ)   ∈ [-1, 1]
```

---

## 4️⃣ Information Theory

```
Entropy:              H(X) = -Σ P(x) log₂ P(x)
                      (higher = more uncertain/informative)

Joint Entropy:        H(X,Y) = -Σ P(x,y) log P(x,y)

Conditional Entropy:  H(Y|X) = H(X,Y) - H(X)

Information Gain:     IG(Y|X) = H(Y) - H(Y|X)
                      (used in Decision Trees)

Gini Impurity:        G = 1 - Σ pᵢ²

KL Divergence:        DKL(P||Q) = Σ P(x) log(P(x)/Q(x))
                      (not symmetric; always ≥ 0)

Cross-Entropy:        H(P,Q) = -Σ P(x) log Q(x)
                      = H(P) + DKL(P||Q)
```

---

## 5️⃣ Loss Functions (Formal)

```
MSE:           L = (1/n) Σ(yᵢ - ŷᵢ)²
MAE:           L = (1/n) Σ|yᵢ - ŷᵢ|
Huber:         L = { ½(y-ŷ)²           if |y-ŷ| ≤ δ
                   { δ|y-ŷ| - ½δ²      otherwise

Log Loss:      L = -(1/n) Σ[yᵢ log(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ)]
               (binary cross-entropy)

Hinge Loss:    L = max(0, 1 - yᵢ · ŷᵢ)   (SVM)

Softmax:       σ(zᵢ) = eᶻⁱ / Σⱼ eᶻʲ
```

---

## 6️⃣ Model Evaluation Formulas

```
Accuracy    = (TP + TN) / (TP + FP + FN + TN)
Precision   = TP / (TP + FP)
Recall      = TP / (TP + FN)
F1          = 2 × (Precision × Recall) / (Precision + Recall)
Fβ          = (1+β²) × (P × R) / (β²P + R)

TPR (Sensitivity/Recall) = TP / (TP + FN)
FPR (Fall-out)           = FP / (FP + TN)
Specificity              = TN / (TN + FP) = 1 - FPR

R²          = 1 - SS_res/SS_tot = 1 - Σ(yᵢ-ŷᵢ)²/Σ(yᵢ-ȳ)²
Adj R²      = 1 - (1-R²)(n-1)/(n-p-1)   p = # features

MAPE        = (100/n) Σ|(yᵢ-ŷᵢ)/yᵢ|
```

---

## 7️⃣ Regularization Math

```
Ridge (L2):     L_total = MSE + λ Σwᵢ²
                Closed form: w = (XᵀX + λI)⁻¹Xᵀy

Lasso (L1):     L_total = MSE + λ Σ|wᵢ|
                No closed form; use coordinate descent

Elastic Net:    L_total = MSE + λ₁Σ|wᵢ| + λ₂Σwᵢ²
```

---

## 8️⃣ Activation Functions (Formal)

```
Sigmoid:     σ(x) = 1/(1+e⁻ˣ)          σ'(x) = σ(x)(1-σ(x))
Tanh:        tanh(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)    tanh'(x) = 1-tanh²(x)
ReLU:        max(0, x)                   derivative: 0 if x<0, 1 if x>0
Leaky ReLU:  max(αx, x)  α≈0.01
ELU:         x if x≥0, α(eˣ-1) if x<0
Softmax:     σ(zᵢ) = eᶻⁱ / Σⱼ eᶻʲ
GELU:        x · Φ(x)  where Φ is standard normal CDF
```

---

## 9️⃣ Deep Learning

### Attention
```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V

Multi-Head:
  head_i = Attention(QWᵢQ, KWᵢK, VWᵢV)
  MultiHead = Concat(head₁,...,headₕ) · Wᴼ
```

### Batch Normalization
```
μ_B = (1/m) Σ xᵢ
σ²_B = (1/m) Σ(xᵢ - μ_B)²
x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)
yᵢ = γ x̂ᵢ + β    ← learned scale and shift
```

### Weight Initialization
```
Xavier (Glorot):   w ~ U[-√(6/(nᵢₙ+nₒᵤₜ)), √(6/(nᵢₙ+nₒᵤₜ))]
He (Kaiming):      w ~ N(0, √(2/nᵢₙ))   ← for ReLU
```

---

## 🔟 Dimensionality Reduction

### PCA
```
1. Center data:  X_c = X - μ
2. Covariance:   C = (1/n) X_cᵀ X_c
3. Eigen decomp: C = QΛQᵀ
4. Project:      Z = X_c Q_k   (top k eigenvectors)

Variance explained by component i:  λᵢ / Σ λⱼ
```

### t-SNE
```
Similarity in high-dim:  p_{j|i} = exp(-||xᵢ-xⱼ||²/2σᵢ²) / Σ_{k≠i} exp(-||xᵢ-xₖ||²/2σᵢ²)
Similarity in low-dim:   q_{ij} = (1 + ||yᵢ-yⱼ||²)⁻¹ / Σ_{k≠l} (1 + ||yₖ-yₗ||²)⁻¹
KL divergence:           KL(P||Q) = Σ pᵢⱼ log(pᵢⱼ/qᵢⱼ)
```

---

*Part of Complete ML Notes Repository*
