#  Key Formulas — Interview Quick Reference

> All the formulas you need to know by heart. No derivations here — just the essentials.

---

##  Evaluation Metrics

```
━━━━━━━━ CLASSIFICATION ━━━━━━━━

Accuracy    = (TP + TN) / (TP + FP + FN + TN)
Precision   = TP / (TP + FP)
Recall      = TP / (TP + FN)
F1          = 2 × Precision × Recall / (Precision + Recall)
Fβ          = (1+β²) × P × R / (β²P + R)     β>1 favors recall, β<1 favors precision

Specificity = TN / (TN + FP)      ← True Negative Rate
FPR         = FP / (FP + TN)      ← x-axis of ROC curve
TPR         = TP / (TP + FN)      ← y-axis of ROC curve = Recall

━━━━━━━━ REGRESSION ━━━━━━━━

MAE   = (1/n) Σ |yᵢ - ŷᵢ|
MSE   = (1/n) Σ (yᵢ - ŷᵢ)²
RMSE  = √MSE
MAPE  = (100/n) Σ |yᵢ-ŷᵢ|/yᵢ

R²    = 1 - SS_res / SS_tot
      = 1 - Σ(yᵢ-ŷᵢ)² / Σ(yᵢ-ȳ)²

Adj R² = 1 - (1-R²)(n-1) / (n-p-1)      p = number of features
```

---

##  Loss Functions

```
MSE:         L = (1/n) Σ (yᵢ - ŷᵢ)²
MAE:         L = (1/n) Σ |yᵢ - ŷᵢ|
Log Loss:    L = -(1/n) Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
Hinge:       L = (1/n) Σ max(0, 1 - yᵢ ŷᵢ)
Cross-Entropy: L = -Σ yᵢ log(ŷᵢ)        (multiclass)
```

---

##  Core Algorithms

```
━━━━━━━━ LINEAR REGRESSION ━━━━━━━━
ŷ = Xw + b
Loss: MSE = (1/n)||y - Xw||²
Closed form: w = (XᵀX)⁻¹ Xᵀy

━━━━━━━━ LOGISTIC REGRESSION ━━━━━━━━
z = wᵀx + b
P(y=1|x) = σ(z) = 1/(1 + e⁻ᶻ)
Loss: -[y log(ŷ) + (1-y) log(1-ŷ)]

━━━━━━━━ REGULARIZATION ━━━━━━━━
Ridge: L + λ Σwᵢ²
Lasso: L + λ Σ|wᵢ|
Elastic Net: L + λ₁Σ|wᵢ| + λ₂Σwᵢ²

━━━━━━━━ SVM ━━━━━━━━
Hyperplane: wᵀx + b = 0
Margin: 2/||w||
Objective: min ½||w||² subject to yᵢ(wᵀxᵢ+b) ≥ 1
Kernel trick: K(x, x') replaces xᵀx'

━━━━━━━━ NAIVE BAYES ━━━━━━━━
P(y|x₁,...,xₙ) ∝ P(y) Π P(xᵢ|y)

━━━━━━━━ DECISION TREE ━━━━━━━━
Entropy: H = -Σ pᵢ log₂ pᵢ
Gini: G = 1 - Σ pᵢ²
Info Gain: IG = H(parent) - Σ wᵢ H(childᵢ)

━━━━━━━━ K-MEANS ━━━━━━━━
Objective: minimize Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²
Update centroid: μᵢ = (1/|Cᵢ|) Σₓ∈Cᵢ x
```

---

##  Deep Learning

```
━━━━━━━━ NEURON ━━━━━━━━
z = Wx + b
a = g(z)    ← g is activation function

━━━━━━━━ GRADIENT DESCENT ━━━━━━━━
θ ← θ - α ∇_θ L(θ)

━━━━━━━━ BACKPROP ━━━━━━━━
∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W     (chain rule)

━━━━━━━━ ATTENTION ━━━━━━━━
Attention(Q,K,V) = softmax(QKᵀ/√d_k) · V

━━━━━━━━ BATCH NORMALIZATION ━━━━━━━━
μ = (1/m) Σ xᵢ
σ² = (1/m) Σ(xᵢ - μ)²
x̂ = (x - μ) / √(σ² + ε)
y = γx̂ + β

━━━━━━━━ SOFTMAX ━━━━━━━━
σ(zᵢ) = eᶻⁱ / Σⱼ eᶻʲ

━━━━━━━━ DROPOUT (at test time) ━━━━━━━━
Keep neurons ON, multiply weights by (1-p)
OR scale outputs by (1/(1-p)) during training (inverted dropout)
```

---

##  Statistics & Probability

```
━━━━━━━━ BAYES ━━━━━━━━
P(A|B) = P(B|A) · P(A) / P(B)

━━━━━━━━ EXPECTATION ━━━━━━━━
E[X] = Σ x · P(x)            (discrete)
E[X] = ∫ x · f(x) dx         (continuous)

━━━━━━━━ VARIANCE ━━━━━━━━
Var(X) = E[(X-μ)²] = E[X²] - (E[X])²

━━━━━━━━ COVARIANCE ━━━━━━━━
Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)]

━━━━━━━━ CORRELATION ━━━━━━━━
ρ(X,Y) = Cov(X,Y) / (σₓ · σᵧ)    ∈ [-1, 1]

━━━━━━━━ NORMAL DISTRIBUTION ━━━━━━━━
f(x) = (1/√(2πσ²)) · exp(-(x-μ)²/(2σ²))
68% within μ ± σ
95% within μ ± 2σ
99.7% within μ ± 3σ
```

---

##  Information Theory

```
Entropy:       H(X) = -Σ P(x) log₂ P(x)
Cross-Entropy: H(P,Q) = -Σ P(x) log Q(x)
KL Divergence: D_KL(P||Q) = Σ P(x) log(P(x)/Q(x))

Relationship:  H(P,Q) = H(P) + D_KL(P||Q)
               ↑              ↑         ↑
          cross-entropy   true entropy  extra cost
```

---

##  Dimensionality Reduction

```
━━━━━━━━ PCA ━━━━━━━━
1. Covariance matrix: C = (1/(n-1)) XᵀX
2. Eigendecomposition: C = QΛQᵀ
3. Select top k: Z = X · Q_k

Variance explained by PC i: λᵢ / Σ λⱼ

━━━━━━━━ NORMALIZATION ━━━━━━━━
Min-Max:        (x - xmin) / (xmax - xmin)
Z-score:        (x - μ) / σ
Robust:         (x - median) / IQR
```

---

##  Numbers to Know

```
Common defaults:
  k-Fold CV:          k = 5 or 10
  Train/Val/Test:     70/15/15 or 80/10/10
  Random Forest:      sqrt(n_features) features per split
  LR:                 alpha = 0.001 (Adam), 0.01 (SGD)
  Dropout:            p = 0.2 to 0.5
  Batch size:         32, 64, 128, 256

Adam hyperparams:
  β₁ = 0.9   (1st moment decay)
  β₂ = 0.999 (2nd moment decay)
  ε  = 1e-8  (numerical stability)

Statistical significance: p-value < 0.05 (5% significance level)
Correlation: |ρ| > 0.7 = strong, 0.3-0.7 = moderate, <0.3 = weak
VIF > 10 = severe multicollinearity
```

---

*Part of Complete ML Notes Repository | Use this for last-minute interview review*
