# Linear Regression

> **Category:** Supervised Learning
> **Task type:** Regression (predicting continuous values)
> **Prerequisites:** Basic algebra, derivatives, mean/variance
> **Time to study:** 2 hrs for full depth · 15 min for interview prep


---

##  How to use this note

| Your goal | Jump to |
|---|---|
| I want to *understand* it from scratch | Start at §1 → read top to bottom |
| I need the math — intuition + derivations | §4 Math Intuition → §5 Derivation |
| I'm preparing for an *interview* | §10 Key Intuitions + §11 Interview Q&A |
| I want to *implement* it | §12 Scratch + §13 Sklearn Pipeline |
| I want to compare it with other algorithms | §9 Comparison |
| Just a quick refresher | §10 Key Intuitions only |

---

## 1.  The Big Picture

**What problem does this solve?**

Given a set of input variables (features), predict a continuous numeric output. You have historical data — input/output pairs — and you want to learn a mathematical function that maps new inputs to outputs.

**The core idea in one sentence:**
> "Fit the best possible straight line (or hyperplane) through your data points by finding the weights that minimize the sum of squared prediction errors."

**Why does this matter?**

Real-world applications are everywhere: predicting house prices from size and location, forecasting electricity demand from temperature, estimating a patient's blood pressure from clinical measurements, predicting exam scores from hours studied. Any time you need to predict "how much" of something, linear regression is your starting point.

**Where does it fit in the ML landscape?**

```
Supervised Learning
├── Classification  (predict a category: spam/not spam, dog/cat)
└── Regression      (predict a number: price, temperature, score)
    ├── Linear Regression       ← we are here
    ├── Polynomial Regression   (extension of this)
    ├── Ridge / Lasso           (regularized versions)
    └── SVR, Random Forest, XGBoost (non-linear alternatives)
```

---

## 2.  Intuition — Understanding Before Math

### The core analogy

Imagine you're a detective trying to predict house prices. You've seen 1,000 houses sold — you know each house's size (sqft) and its sale price. You want to predict the price of a new house you've never seen.

You notice: bigger houses tend to cost more. If you plotted size on the x-axis and price on the y-axis, the points would form a rough diagonal cloud going up-right. You want to draw a single straight line through that cloud — the line that comes closest to all the points at once.

That line is linear regression. Finding the best line is training. Using the line to predict a new house's price is inference.

**Mapping the analogy to math:**

| What the detective does | What the algorithm does |
|---|---|
| Looks at all past houses | Reads the training set (X, y) |
| Draws a line through the cloud | Learns weights w and bias b |
| Line slope = "price per sqft" | Weight w = coefficient |
| Line y-intercept | Bias b |
| Uses the line to price new house | Computes ŷ = wx + b |
| Adjusts line to fit better | Gradient descent minimizes J |

### Visual intuition — before any formula

```
House price ($)
    │
400k│          ○      ← actual sale price
    │       ○  │ ← residual (error) = y - ŷ
    │    ○  ╱──┘
    │  ○   ╱ ○
300k│ ○   ╱                      ← ŷ = wx + b
    │    ╱  ○                       (the fitted line)
    │   ╱ ○
    │  ╱○
200k│ ╱
    │╱
    └─────────────────────────────── House size (sqft)
         1000     2000     3000

What training does:
  - Start with a random line
  - Measure how wrong each prediction is (residual = y - ŷ)
  - Tilt/shift the line to make residuals smaller overall
  - Repeat until no more improvement

What we're minimizing:
  Not the sum of residuals (they cancel out: positive + negative = 0)
  But the sum of SQUARED residuals — so negatives can't cancel positives
```

### How does it "learn"?

Every iteration, the algorithm measures how wrong it is. The gradient tells it: "tilt the line this way to reduce your total error." It takes a small step in that direction. After thousands of steps, the line settles at the position that minimizes total error. That's the trained model.

---

## 3.  When to Use It

### Use linear regression when:
- [ ] Your output is a continuous numeric value (price, temperature, score)
- [ ] You need the result to be interpretable (each coefficient has meaning)
- [ ] You want a fast, low-compute baseline before trying complex models
- [ ] The relationship between features and target is approximately linear
- [ ] Dataset is small to medium — linear regression is extremely sample-efficient

### Avoid linear regression when:
- [ ] Output is a category → use Logistic Regression or tree-based models
- [ ] Relationship is strongly non-linear → Polynomial Regression, XGBoost, or Neural Net
- [ ] Features are highly correlated (multicollinearity) → Ridge Regression first
- [ ] Dataset has many irrelevant features → Lasso first (for feature selection)

### Data checklist

| Requirement | Details | What breaks if violated |
|---|---|---|
| Target type | Continuous numeric — unbounded | Meaningless predictions if used for classification |
| Feature scaling | Strongly recommended | Unscaled features cause very slow, uneven gradient descent |
| Sample size | Works with as few as 10–20 samples | Coefficients are unreliable with very small n |
| Missing values | Must be handled before fitting | sklearn will throw an error |
| Outlier sensitivity | High — MSE penalizes outliers heavily | Coefficients get pulled toward outliers |
| Multicollinearity | Problem — coefficients become unstable | Use Ridge (L2), VIF analysis, or remove correlated features |

---

## 4.  Mathematical Intuition

> *Andrew Ng teaches derivations step by step. Bishop teaches the probabilistic foundations.*
> *Géron shows how it works in practice. Gilbert Strang gives the geometric view.*
> *fast.ai makes it click through code. ESL gives the statistical theory.*
> *This section synthesizes all of them.*

### Notation

| Symbol | Meaning | Example (house prices) |
|---|---|---|
| `x⁽ⁱ⁾` | Feature vector for example i | Size=1500, Bedrooms=3, Age=10 |
| `y⁽ⁱ⁾` | True output for example i | $320,000 |
| `ŷ⁽ⁱ⁾` | Predicted output for example i | $315,000 |
| `w` | Weight vector (n-dimensional) | [150, 5000, -200] |
| `b` | Bias term | 50000 |
| `m` | Number of training examples | 1000 houses |
| `n` | Number of features | 3 features |
| `X` | Feature matrix, shape (m × n) | 1000 × 3 matrix |
| `J(w,b)` | Cost — average squared error | 4.2 × 10⁹ |
| `α` | Learning rate | 0.01 |

### Math building block 1 — The dot product

The prediction of a linear model is a dot product:

```
ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
  = w · x + b
  = wᵀx + b   (matrix notation)

House example:
  w = [150, 5000, -200]
  x = [1500, 3, 10]    (sqft, bedrooms, age in years)
  b = 50000

  ŷ = 150×1500 + 5000×3 + (-200)×10 + 50000
    = 225000 + 15000 - 2000 + 50000
    = $288,000
```

**Interpretation of weights:** Each weight wᵢ says "if feature xᵢ increases by 1 unit, the prediction changes by wᵢ units, holding all other features fixed." So w₁=150 means "every additional sqft adds $150 to the predicted price."

### Math building block 2 — Partial derivatives

A partial derivative answers: "how does J change if I wiggle just this one weight?"

```
If J(w₁, w₂) = w₁² + 2w₁w₂ + w₂²

∂J/∂w₁ = 2w₁ + 2w₂   (treat w₂ as constant, differentiate w.r.t. w₁)
∂J/∂w₂ = 2w₁ + 2w₂   (treat w₁ as constant, differentiate w.r.t. w₂)

The gradient ∇J = [∂J/∂w₁, ∂J/∂w₂] points uphill.
To minimize J: move w in the direction of -∇J (downhill).
```

**Gradient descent intuition:**

```
   J(w)                   Gradient = slope here (positive → go left)
    │         /
    │        / ← high gradient: far from minimum
    │       /
    │      /
    │     /  ← low gradient: near minimum, take small steps
    │    /____
    │         ← minimum
    └────────────────── w
              w*

Update: w ← w - α × (slope at current w)
```

### Math building block 3 — The chain rule

We need to compute how J depends on w through the intermediate ŷ:

```
J depends on ŷ (which depends on w)
J → ŷ → w

Chain rule: ∂J/∂w = ∂J/∂ŷ × ∂ŷ/∂w
                     ↑         ↑
               how J changes   how ŷ changes
               with ŷ          with w

Concrete example:
  ŷ = w·x + b        so  ∂ŷ/∂w = x
  J = (ŷ - y)²       so  ∂J/∂ŷ = 2(ŷ - y)
  Therefore:
  ∂J/∂w = 2(ŷ - y) × x
```

---

## 5.  Step-by-Step Derivation

> *Full working — no steps skipped. Based on Stanford CS229 notes + Bishop PRML Ch.3.*

### Step 1 — The hypothesis (what the model computes)

**In words:** The model takes your input features, multiplies each by a learned weight, adds a learned bias, and outputs a single number as the prediction.

```
Single example (n features):
  ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
    = wᵀx + b

Vector form (m examples at once):
  ŷ = Xw + b           (matrix multiply: (m×n)·(n,) → (m,))

Where:
  X is the feature matrix (m rows = examples, n columns = features)
  w is the weight vector (n values)
  b is the bias scalar (broadcast to all m predictions)
  ŷ is the prediction vector (m values, one per example)
```

**Why a linear hypothesis?**

Three reasons: (1) computationally cheap — just one dot product, (2) interpretable — each weight has direct meaning, (3) sufficient for many real-world relationships that are approximately linear (or can be made so with feature engineering).

**Sanity check with values:**

```
If w = [2], b = 1, and x = [3]:
  ŷ = 2×3 + 1 = 7  ✓ (linear: increase x by 1 → ŷ increases by w=2)

If w = [0, 0], b = 5:
  ŷ = 5 for any x  (weights=0 → model predicts constant = bias only)
```

---

### Step 2 — The loss function (single example)

**In words:** For one training example, how wrong is our prediction? We need a number that is 0 when ŷ = y, and grows as the error grows.

**Why Mean Squared Error (MSE)?** Three candidates:

```
Option A: Raw error = (y - ŷ)
  Problem: positive and negative errors cancel.
  Example: predict 10 instead of 12 (error = -2)
           predict 14 instead of 12 (error = +2)
           Total = 0. Model looks perfect. It isn't.

Option B: Absolute error = |y - ŷ|
  Better: no cancellation. But not differentiable at 0 (gradient descent breaks).

Option C: Squared error = (y - ŷ)²  ← we use this
  Best for gradient descent: smooth, differentiable everywhere, penalizes
  large errors more (2× error → 4× penalty), always non-negative.
```

**Single-example loss:**

```
L(ŷ, y) = (y - ŷ)²  = (y - wᵀx - b)²

Or equivalently: (ŷ - y)²   ← sign doesn't matter since we square it

Verify:
  Perfect:  ŷ = y   → L = (y-y)² = 0  ✓
  Off by 5: ŷ = y+5 → L = 25          ✓ (positive, not zero)
  Off by 10:ŷ = y+10→ L = 100         ✓ (4× bigger for 2× bigger error)
```

---

### Step 3 — The cost function (all examples)

**In words:** Average the loss over all m training examples. This gives one number summarizing how wrong the model is overall.

```
J(w, b) = (1/2m) Σᵢ₌₁ᵐ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²
         = (1/2m) Σᵢ₌₁ᵐ (wᵀx⁽ⁱ⁾ + b - y⁽ⁱ⁾)²

Note: The (1/2) is conventional — it makes the derivative cleaner
      (the 2 from squaring cancels with 1/2). Doesn't change the minimum.

Matrix form (compact and fast to compute):
  J(w, b) = (1/2m) ‖Xw + b - y‖²
           = (1/2m) (Xw + b - y)ᵀ(Xw + b - y)
```

**What does J look like as a function of w?**

```
For 1 parameter (w):                For 2 parameters (w₁, w₂):

  J(w)                                    J(w₁,w₂)
    │    ╲     ╱                     ┌──────────────────┐
    │     ╲   ╱                      │  level curves     │
    │      ╲ ╱                       │    (ellipses)     │
    │       V                        │        ●w*        │
    │                                │                   │
    └──────────────── w              └──────────────────┘
         ↑ w* (bowl bottom)          Gradient points toward center
         
BOTH are convex bowls → gradient descent is GUARANTEED to find the global minimum.
This is a big deal — many ML loss surfaces are not convex.
```

---

### Step 4 — Compute gradients (full derivation, no shortcuts)

**Gradient w.r.t. w (∂J/∂w):**

```
J(w, b) = (1/2m) Σᵢ (wᵀx⁽ⁱ⁾ + b - y⁽ⁱ⁾)²

Step 1: Pull out the constant (1/2m) and bring ∂/∂w inside the sum
  ∂J/∂w = (1/2m) Σᵢ ∂/∂w [(wᵀx⁽ⁱ⁾ + b - y⁽ⁱ⁾)²]

Step 2: Apply chain rule. Let eᵢ = (wᵀx⁽ⁱ⁾ + b - y⁽ⁱ⁾) = "error on example i"
  ∂/∂w [eᵢ²] = 2eᵢ · ∂eᵢ/∂w     ← chain rule: d(u²)/dw = 2u · du/dw

Step 3: Compute ∂eᵢ/∂w
  eᵢ = wᵀx⁽ⁱ⁾ + b - y⁽ⁱ⁾
  ∂eᵢ/∂w = x⁽ⁱ⁾                   ← x⁽ⁱ⁾ is the coefficient of w

Step 4: Combine
  ∂J/∂w = (1/2m) Σᵢ 2eᵢ · x⁽ⁱ⁾
         = (1/m) Σᵢ eᵢ · x⁽ⁱ⁾
         = (1/m) Σᵢ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾) · x⁽ⁱ⁾

Matrix form (compact):
  ∂J/∂w = (1/m) Xᵀ(ŷ - y)   ← this is what we implement in code
           (n×m) · (m×1) = (n×1) ✓ (gradient has same shape as w)
```

**Gradient w.r.t. b (∂J/∂b):**

```
Same chain rule, but ∂eᵢ/∂b = 1 (since b has coefficient 1 in eᵢ)

∂J/∂b = (1/2m) Σᵢ 2eᵢ · 1
       = (1/m) Σᵢ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)

Intuition: the bias gradient is just the mean prediction error.
If your model consistently predicts too high: ∂J/∂b > 0 → b decreases. ✓
```

**Sanity check the gradients:**

```
If ŷ > y (predicting too high):
  (ŷ - y) > 0
  ∂J/∂w = (1/m) Xᵀ(ŷ-y) > 0 (assuming x > 0)
  Update: w ← w - α·(positive) → w decreases
  Effect: smaller w → smaller ŷ → closer to y  ✓

If ŷ < y (predicting too low):
  (ŷ - y) < 0
  ∂J/∂w < 0
  Update: w ← w - α·(negative) → w increases
  Effect: larger w → larger ŷ → closer to y  ✓
```

---

### Step 5 — The update rule (gradient descent)

```
Initialize: w = 0 (or random), b = 0

Repeat for T iterations {

    # Forward pass: compute predictions
    ŷ = Xw + b                         # shape: (m,)

    # Compute gradients
    dw = (1/m) Xᵀ(ŷ - y)              # shape: (n,)
    db = (1/m) sum(ŷ - y)              # scalar

    # Update parameters simultaneously
    w ← w - α · dw
    b ← b - α · db
}

CRITICAL: Update w AND b using the OLD gradients.
Don't update w first, then use the NEW w to compute db.
Both updates use gradients from the same forward pass.
```

**Closed-form solution (Normal Equation) — when m is small:**

```
Instead of gradient descent, solve analytically:

∂J/∂w = 0  →  (1/m)Xᵀ(Xw - y) = 0
              →  XᵀXw = Xᵀy
              →  w* = (XᵀX)⁻¹ Xᵀy    ← Normal Equation

Pros: exact solution in one step, no learning rate to tune
Cons: O(n³) — very slow for n > 10,000 features
      XᵀX must be invertible (fails with multicollinearity)

Rule of thumb:
  n < 10,000 features → Normal Equation (exact, fast)
  n ≥ 10,000 features → Gradient Descent (iterative, scalable)
```

---

## 6.  Probabilistic Interpretation

> *Why is MSE the right loss? Where does it come from?*
> *Source: Andrew Ng CS229 notes, Bishop PRML Ch.1–3*

**Assume:** The true relationship is linear, but with Gaussian noise:

```
y⁽ⁱ⁾ = wᵀx⁽ⁱ⁾ + b + εᵢ,    εᵢ ~ N(0, σ²)

This means, given x, the output y is normally distributed:
  y | x; w, b  ~  N(wᵀx + b,  σ²)

PDF: P(y|x; w, b) = (1/√(2πσ²)) exp(-(y - wᵀx - b)² / 2σ²)
```

**Maximum Likelihood Estimation:**

```
We want to find w, b that makes our observed data most probable:

L(w, b) = Πᵢ P(y⁽ⁱ⁾ | x⁽ⁱ⁾; w, b)             ← likelihood (product: independent)

log L(w, b) = Σᵢ log P(y⁽ⁱ⁾ | x⁽ⁱ⁾; w, b)      ← log-likelihood (easier to work with)

           = Σᵢ log [ (1/√(2πσ²)) exp(-(y⁽ⁱ⁾ - ŷ⁽ⁱ⁾)²/2σ²) ]

           = Σᵢ [ -½log(2πσ²) - (y⁽ⁱ⁾ - ŷ⁽ⁱ⁾)²/2σ² ]

           = -m/2 · log(2πσ²) - (1/2σ²) Σᵢ (y⁽ⁱ⁾ - ŷ⁽ⁱ⁾)²
                                          ↑
                             This is -(constant) × (MSE cost)

Maximizing log L(w,b) = Minimizing Σᵢ (y⁽ⁱ⁾ - ŷ⁽ⁱ⁾)² = Minimizing MSE
```

**The key insight:** MSE is not an arbitrary choice. It is exactly what Maximum Likelihood prescribes when you assume Gaussian noise. If you assumed different noise (e.g., Laplacian), you'd get a different loss (MAE). The loss function encodes your belief about the noise structure.

---

## 7.  Geometric Interpretation

> *Three ways to see linear regression geometrically.*
> *Source: Gilbert Strang's Linear Algebra, ESL Ch.3*

### View 1 — Fitting a hyperplane to a point cloud

```
2D (one feature):     ŷ = wx + b  →  a line
3D (two features):    ŷ = w₁x₁ + w₂x₂ + b  →  a plane
n+1 D (n features):   ŷ = wᵀx + b  →  a hyperplane

Training = finding the hyperplane that minimizes squared distances
           from each data point to the hyperplane (measured vertically)
```

### View 2 — Projection (linear algebra view)

```
The columns of X span a subspace of ℝᵐ.
The target vector y lives in ℝᵐ.
The best prediction ŷ = Xw is the PROJECTION of y onto the column space of X.

        y (true target, m-dim vector)
        │╲
        │ ╲  ← residual vector (y - ŷ), perpendicular to column space
        │  ╲
        │   ŷ = Xw (projection)
        │    ────────────── Column space of X

The Normal Equation XᵀXw = Xᵀy says exactly this:
  Xᵀ(y - Xw) = 0  ←  "residuals are perpendicular to column space"
```

### View 3 — Gradient descent as a ball rolling downhill

```
The loss surface J(w₁, w₂) is a bowl (for MSE):

    J
    │    ╲_____
    │         ╲___
    │              ╲___ ← each gradient step rolls the ball toward the bottom
    │                  ╲
    │                   ● minimum (w₁*, w₂*)
    └────────────────────────── w
    
Learning rate controls step size:
  α too big: ball overshoots, bounces between walls → diverge
  α too small: ball barely moves → training takes forever
  α just right: smooth descent to the minimum
```

---

## 8.  Assumptions — When Linear Regression Breaks

| Assumption | What it means | How to detect | Consequence if violated | Fix |
|---|---|---|---|---|
| Linearity | True relationship is linear in features | Plot residuals vs ŷ — should be random scatter | Systematic under/overprediction | Polynomial features, log transforms |
| Independence | Examples are independent (no autocorrelation) | Durbin-Watson test, plot residuals vs time | Underestimated std errors | Time series models (ARIMA), GLS |
| Homoscedasticity | Variance of errors is constant | Plot residuals vs ŷ — no funnel shape | Inefficient estimates, biased std errors | Log-transform target, WLS |
| Normality of errors | Residuals are normally distributed | Q-Q plot, Shapiro-Wilk test | Inference (CI, p-values) is unreliable | Not critical for large n (CLT helps) |
| No multicollinearity | Features are not correlated with each other | Variance Inflation Factor (VIF > 10) | Unstable, uninterpretable coefficients | Ridge Regression, remove correlated features |
| No outliers | No extreme data points | Cook's distance, leverage plot | Coefficients heavily distorted | Robust regression (Huber loss), remove/cap outliers |

**Diagnostic code:**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def diagnostic_plots(y_true, y_pred, feature_matrix=None):
    """
    Run the 4 core diagnostic plots for linear regression.
    These tell you whether your assumptions hold.
    """
    residuals = y_true - y_pred
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Residuals vs Fitted: check linearity + homoscedasticity
    axes[0,0].scatter(y_pred, residuals, alpha=0.4, s=20)
    axes[0,0].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[0,0].set_xlabel('Fitted values'); axes[0,0].set_ylabel('Residuals')
    axes[0,0].set_title('Residuals vs Fitted\n(should be random scatter around 0)')
    # Pattern → non-linearity. Funnel → heteroscedasticity.

    # 2. Q-Q plot: check normality of residuals
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals)
    axes[0,1].plot(osm, osr, 'o', alpha=0.4, markersize=3)
    axes[0,1].plot(osm, slope*np.array(osm)+intercept, 'r--', linewidth=1)
    axes[0,1].set_xlabel('Theoretical quantiles'); axes[0,1].set_ylabel('Sample quantiles')
    axes[0,1].set_title('Normal Q-Q Plot\n(should follow the line)')

    # 3. Scale-location: check homoscedasticity
    sqrt_abs_res = np.sqrt(np.abs(residuals))
    axes[1,0].scatter(y_pred, sqrt_abs_res, alpha=0.4, s=20)
    axes[1,0].set_xlabel('Fitted values'); axes[1,0].set_ylabel('√|Residuals|')
    axes[1,0].set_title('Scale-Location\n(should be flat/horizontal)')

    # 4. Actual vs Predicted: sanity check
    axes[1,1].scatter(y_true, y_pred, alpha=0.4, s=20)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1,1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
    axes[1,1].set_xlabel('Actual'); axes[1,1].set_ylabel('Predicted')
    axes[1,1].set_title('Actual vs Predicted\n(should hug the diagonal)')

    plt.suptitle('Linear Regression Diagnostic Plots', fontsize=14)
    plt.tight_layout()
    plt.show()
```

---

## 9.  Algorithm Comparison

| Dimension | Linear Regression | Ridge (L2) | Lasso (L1) | Random Forest | XGBoost |
|---|---|---|---|---|---|
| Handles non-linearity | No | No | No | Yes | Yes |
| Interpretability | High | High | High | Low | Low |
| Training speed | Very fast | Very fast | Fast | Slow | Slow |
| Needs feature scaling | Yes | Yes | Yes | No | No |
| Feature selection | No | No | Yes (via zeroed weights) | Feature importance | Feature importance |
| Handles multicollinearity | Poorly | Well | Picks one | Well | Well |
| Overfitting risk | Low | Lower | Lower | Medium | High |
| When to use | Baseline, interpretability | Correlated features | Many irrelevant features | Non-linear patterns | Best accuracy |

**When regularization changes the algorithm:**

```
No reg:  J = (1/2m)Σ(ŷ-y)²              → Ordinary Least Squares
Ridge:   J = (1/2m)Σ(ŷ-y)² + (λ/2m)Σwᵢ² → Shrinks all weights toward 0
Lasso:   J = (1/2m)Σ(ŷ-y)² + (λ/m)Σ|wᵢ| → Drives some weights to exactly 0

Gradient with L2:  ∂J/∂w = (1/m)Xᵀ(ŷ-y) + (λ/m)w
Update with L2:    w ← w - α·[(1/m)Xᵀ(ŷ-y) + (λ/m)w]
                     = w·(1 - αλ/m) - α·(1/m)Xᵀ(ŷ-y)
                          ↑
                     "weight decay" — w shrinks every step by factor (1-αλ/m)
```

**Geometry of regularization — why L1 gives sparse solutions:**

```
Constraint view (equivalent to adding penalty to J):

L2 constraint (Ridge):      L1 constraint (Lasso):
   w₂                           w₂
    │    ●                        │  ●
    │   ╱ ← elliptical           │ ╱ ← elliptical
    │  ╱   contours of J         │╱   contours of J
    │ ╱                          │
    │╱_____ ← L2 ball            ●_____ ← L1 diamond
    │       (smooth circle)      │     (sharp corners!)
    └────────────── w₁           └───────────── w₁

L2: ellipse touches circle on curved surface → solution has wᵢ ≠ 0 (just small)
L1: ellipse usually touches diamond at a CORNER (where one axis = 0) → sparse solution!
```

---

## 10.  Key Intuitions to Remember

> *These are the things you want to be able to recall in 5 seconds.*

**1. Linear regression finds the least-squares line (hyperplane)**
> You're minimizing the sum of squared vertical distances from data points to your predicted line. "Least squares" is the oldest and most natural definition of "best fit" for a line. Everything else — gradients, updates, the Normal Equation — is just machinery to find it efficiently.

**2. MSE is Gaussian MLE in disguise**
> The MSE loss function is not arbitrary. It's exactly what you get when you assume your data has Gaussian noise and you do Maximum Likelihood Estimation. If your noise is Laplacian, use MAE instead. The noise model you believe in dictates the loss function.

**3. The gradient is just the average error times the feature**
> ∂J/∂w = (1/m)Xᵀ(ŷ-y). Unpack this: for each weight wᵢ, its gradient is the average of (prediction error × xᵢ). When the model systematically overpredicts, this gradient is positive, so wᵢ decreases. Clean feedback loop.

**4. Scaling features is not optional — it's required**
> Unscaled features make the loss surface a very elongated ellipse (one axis is much longer). Gradient descent bounces slowly along the narrow dimension. Scale your features first, and the surface becomes a nice circle → fast, direct convergence.

**5. Linear regression is a projection, not just a line**
> Geometrically, ŷ = Xw is the projection of y onto the column space of X. The residuals y - ŷ are perpendicular to that space. This is why the Normal Equation works: XᵀXw = Xᵀy literally encodes the perpendicularity condition.

### The 30-second mental model

```
Data: X (features), y (targets)
      ↓
Model: ŷ = Xw + b   ← linear combination of features
      ↓
Error: J = (1/2m)‖ŷ - y‖²  ← MSE cost (convex bowl)
      ↓
Gradient: ∂J/∂w = (1/m)Xᵀ(ŷ-y)  ← how to improve w
      ↓
Update: w ← w - α·∂J/∂w   ← move downhill on the cost surface
      ↓
Repeat → w converges to the weights that minimize prediction error
```

### What surprises people about linear regression

- "Linear Regression can model non-linear relationships!" — Yes, if you add polynomial or interaction features. The model is linear in the *weights*, not necessarily in the *features*.
- "More features can make linear regression worse" — Yes, multicollinearity makes weights unstable and the model fits noise. Feature selection or Ridge helps.
- "Linear regression can be derived from probability theory" — Yes. MSE is Gaussian MLE. This makes linear regression a principled probabilistic model, not just a geometric trick.

---

## 11.  Interview Q&A

### Fundamental understanding

---

**Q: "Explain linear regression to me."**

> **Model answer:** Linear regression learns a weighted sum of features to predict a continuous target. Mathematically, ŷ = wᵀx + b. We find the weights w and bias b by minimizing Mean Squared Error — the average squared difference between predictions and true values. We do this with gradient descent: iteratively compute the gradient of the cost and step the weights in the opposite direction.
>
> **Key signal:** Clean definition + mention of optimization. Don't just say "it fits a line."

---

**Q: "Why do we use MSE and not MAE?"**

> **Model answer:** Both are valid, but MSE has two advantages: (1) it's differentiable everywhere — MAE has a kink at 0 that breaks gradient descent, (2) it comes directly from Maximum Likelihood Estimation under Gaussian noise assumptions. The downside is MSE is more sensitive to outliers — one large error gets squared and dominates the loss. If your data has many outliers, MAE or Huber loss is better.
>
> **Key signal:** Probabilistic motivation (MLE) + practical trade-off awareness.

---

**Q: "What's the difference between gradient descent and the Normal Equation?"**

> **Model answer:** Both find the same optimal weights, but differently. The Normal Equation solves for w* = (XᵀX)⁻¹Xᵀy directly — one computation, exact solution, but O(n³) for n features. Gradient descent is iterative — it takes many small steps downhill on the loss surface, scales to large n (millions of features), but needs a learning rate and convergence check. For n < ~10,000 features, use Normal Equation. For larger n, use gradient descent.
>
> **Key signal:** Knows both, knows trade-offs, knows when to use each.

---

**Q: "Your model has very different coefficients each time you retrain on similar data. What's happening?"**

> **Model answer:** This is the signature of multicollinearity — correlated features. When two features are correlated, the model has many equally good ways to split the coefficient between them, so small data perturbations cause large coefficient swings. Fix: check VIF scores, remove correlated features, or switch to Ridge Regression (L2 regularization), which stabilizes coefficients by penalizing large values.
>
> **Key signal:** Knows the diagnosis *and* the fix. Not just "correlation is bad."

---

### Mathematical depth

---

**Q: "Derive the gradient of MSE with respect to w."**

> **Answer (say step by step):**
> ```
> J(w,b) = (1/2m) Σᵢ (wᵀx⁽ⁱ⁾ + b - y⁽ⁱ⁾)²
>
> ∂J/∂w = (1/2m) · 2 · Σᵢ (wᵀx⁽ⁱ⁾ + b - y⁽ⁱ⁾) · x⁽ⁱ⁾   [chain rule]
>        = (1/m) Σᵢ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾) · x⁽ⁱ⁾
>
> In matrix form: (1/m) Xᵀ(ŷ - y)
> ```
> **Key signal:** Shows the chain rule step. Not just "the answer is Xᵀ(ŷ-y)."

---

**Q: "What is the probabilistic justification for linear regression?"**

> **Model answer:** Assume y = wᵀx + b + ε where ε ~ N(0, σ²). Then P(y|x;w) is Gaussian. The likelihood of the training set is the product of these Gaussians. Taking the log, maximizing log-likelihood equals minimizing the sum of squared errors. So MSE linear regression is exactly Gaussian MLE — the loss function is not an arbitrary choice, it encodes our probabilistic belief about the noise.
>
> **Key signal:** Can connect loss → noise model → MLE. Senior-level thinking.

---

### Trap questions

---

**Q: "R² is 0.95. Is your model good?"**

> **Model answer:** Not necessarily. R² only tells you how much variance in y the model explains — it doesn't tell you if the model is appropriate. I'd also check: (1) residual plots — are there patterns? (2) actual vs predicted plot — any systematic bias? (3) performance on held-out test data — train R² can be high due to overfitting. (4) does the model make sense — are the coefficients in the right direction and magnitude?

---

**Q: "Can linear regression handle categorical features?"**

> **Model answer:** Yes, but they need to be encoded first. One-hot encoding converts a categorical feature with k levels into k-1 binary columns (drop one to avoid the dummy variable trap — perfect multicollinearity). Binary features (0/1) work directly. Ordinal features can sometimes be left as integers if the ordering is meaningful.

---

## 12.  From-Scratch Implementation (NumPy)

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionScratch:
    """
    Linear Regression from scratch using NumPy.

    Mathematical summary (matches derivation in §5 above):
        Hypothesis:  ŷ = Xw + b
        Loss:        J = (1/2m) ‖ŷ - y‖²
        Gradients:   ∂J/∂w = (1/m) Xᵀ(ŷ - y)
                     ∂J/∂b = (1/m) Σ(ŷ - y)
        Updates:     w ← w - α·∂J/∂w
                     b ← b - α·∂J/∂b
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000,
                 regularization=None, lambda_=0.1):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.reg = regularization      # None, 'l1', 'l2'
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _forward(self, X):
        """Hypothesis: ŷ = Xw + b"""
        return X @ self.weights + self.bias  # (m,n)·(n,) + scalar → (m,)

    def _cost(self, y_pred, y_true):
        """J = (1/2m) Σ(ŷ - y)² + regularization"""
        m = len(y_true)
        mse = (1 / (2*m)) * np.sum((y_pred - y_true)**2)
        if self.reg == 'l2':
            mse += (self.lambda_ / (2*m)) * np.sum(self.weights**2)
        elif self.reg == 'l1':
            mse += (self.lambda_ / m) * np.sum(np.abs(self.weights))
        return mse

    def _gradients(self, X, y_pred, y_true):
        """
        ∂J/∂w = (1/m) Xᵀ(ŷ - y)  [from §5 Step 4]
        ∂J/∂b = (1/m) Σ(ŷ - y)
        """
        m = len(y_true)
        error = y_pred - y_true       # (m,)
        dw = (1/m) * X.T @ error      # (n,m)·(m,) → (n,)
        db = (1/m) * np.sum(error)    # scalar

        if self.reg == 'l2':
            dw += (self.lambda_ / m) * self.weights
        elif self.reg == 'l1':
            dw += (self.lambda_ / m) * np.sign(self.weights)
        return dw, db

    def fit(self, X, y, verbose=True):
        X, y = np.array(X, float), np.array(y, float)
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        self.loss_history = []

        for i in range(self.n_iter):
            y_pred = self._forward(X)
            loss = self._cost(y_pred, y)
            self.loss_history.append(loss)
            dw, db = self._gradients(X, y_pred, y)
            self.weights -= self.lr * dw
            self.bias    -= self.lr * db

            if verbose and i % (self.n_iter // 5) == 0:
                print(f"  Iter {i:5d} | Loss: {loss:.6f} | "
                      f"w: {self.weights.round(3)} | b: {self.bias:.3f}")
        return self

    def predict(self, X):
        return self._forward(np.array(X, float))

    def score(self, X, y):
        """R² score: 1.0 = perfect, 0.0 = predict mean, <0 = worse than mean"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        return 1 - ss_res / ss_tot

    def plot_loss_curve(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.loss_history, linewidth=1.5, color='#378ADD')
        plt.xlabel('Iteration'); plt.ylabel('Cost J(w,b)')
        plt.title('Training loss curve (log scale)')
        plt.yscale('log'); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()
        print(f"Initial loss: {self.loss_history[0]:.4f}")
        print(f"Final loss:   {self.loss_history[-1]:.4f}")

    def normal_equation(self, X, y):
        """
        Exact solution: w* = (XᵀX)⁻¹ Xᵀy
        Use when n < 10,000. Skips gradient descent entirely.
        """
        X_b = np.c_[np.ones(len(X)), X]   # add bias column
        w_full = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y  # pinv handles singular cases
        self.bias = w_full[0]
        self.weights = w_full[1:]
        return self


# ── Test: verify scratch = sklearn ───────────────────────────────────────────

np.random.seed(42)
m = 300
X = np.random.randn(m, 3)
true_w = np.array([2.5, -1.0, 0.5])
true_b = 3.0
y = X @ true_w + true_b + np.random.randn(m) * 0.5

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gradient descent version
gd_model = LinearRegressionScratch(learning_rate=0.1, n_iterations=500)
gd_model.fit(X_train, y_train, verbose=False)
gd_model.plot_loss_curve()

# Normal Equation version (should match gradient descent)
ne_model = LinearRegressionScratch()
ne_model.normal_equation(X_train, y_train)

# Sklearn reference
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
sk = LinearRegression().fit(X_train, y_train)

print("\nParameter recovery:")
print(f"True weights:    {true_w} | bias: {true_b}")
print(f"GD weights:      {gd_model.weights.round(3)} | bias: {gd_model.bias:.3f}")
print(f"NE weights:      {ne_model.weights.round(3)} | bias: {ne_model.bias:.3f}")
print(f"Sklearn weights: {sk.coef_.round(3)} | bias: {sk.intercept_:.3f}")

print("\nTest R² scores:")
print(f"GD model:  {gd_model.score(X_test, y_test):.4f}")
print(f"NE model:  {ne_model.score(X_test, y_test):.4f}")
print(f"Sklearn:   {r2_score(y_test, sk.predict(X_test)):.4f}")
# All three should be nearly identical
```

---

## 13.  Production-Ready Sklearn Pipeline

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── 1. Load and inspect ────────────────────────────────────────────────────

# df = pd.read_csv('your_data.csv')
# X = df.drop('target', axis=1)
# y = df['target']

# Before any modeling, always check:
print(f"Shape: {X.shape}")
print(f"Missing values: {X.isnull().sum().sum()}")
print(f"Target range: [{y.min():.2f}, {y.max():.2f}], mean: {y.mean():.2f}")
print(f"Target skew: {y.skew():.2f}  (>1 or <-1 → consider log transform)")

# If target is right-skewed (common for prices), log-transform it:
# y = np.log1p(y)  # predict log(price), then exponentiate to get price back

# ── 2. Split FIRST — never touch test set during preprocessing ─────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 3. Build pipeline ─────────────────────────────────────────────────────

# Option A: Standard linear regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),      # z-score normalization (fit on train only)
    ('model', Ridge(alpha=1.0))        # Ridge by default — safer than plain LR
])

# Option B: Non-linear features (polynomial regression)
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('poly', PolynomialFeatures(degree=2, include_bias=False)),
#     ('model', Ridge(alpha=1.0))
# ])

# ── 4. Cross-validation ────────────────────────────────────────────────────

cv_scores = cross_val_score(
    pipeline, X_train, y_train,
    cv=5, scoring='r2', n_jobs=-1
)
print(f"\nCV R² scores: {cv_scores.round(4)}")
print(f"Mean R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 5. Hyperparameter tuning ──────────────────────────────────────────────

param_grid = {
    'model': [Ridge(), Lasso(), ElasticNet()],
    'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
}

gs = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, refit=True)
gs.fit(X_train, y_train)

print(f"\nBest model: {gs.best_params_['model'].__class__.__name__}")
print(f"Best alpha: {gs.best_params_['model__alpha']}")
print(f"Best CV R²: {gs.best_score_:.4f}")

# ── 6. Final evaluation ────────────────────────────────────────────────────

best_model = gs.best_estimator_
y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"\nTest set performance:")
print(f"  RMSE: {rmse:.4f}  (same units as target — lower is better)")
print(f"  MAE:  {mae:.4f}   (more robust to outliers than RMSE)")
print(f"  R²:   {r2:.4f}   (1.0 = perfect, 0.0 = predict mean)")

# ── 7. Diagnostic plots ────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Learning curves — diagnose bias vs variance
sizes, tr_scores, va_scores = learning_curve(
    best_model, X_train, y_train,
    cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='r2', n_jobs=-1
)
axes[0].plot(sizes, tr_scores.mean(axis=1), 'o-', label='Train R²')
axes[0].plot(sizes, va_scores.mean(axis=1), 'o-', label='Validation R²')
axes[0].fill_between(sizes, va_scores.mean(1)-va_scores.std(1),
                     va_scores.mean(1)+va_scores.std(1), alpha=0.1)
axes[0].set_xlabel('Training size'); axes[0].set_ylabel('R²')
axes[0].set_title('Learning curves\n(gap = overfitting, both low = underfitting)')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Plot 2: Residuals — check for patterns (should be random)
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Residual (actual - predicted)')
axes[1].set_title('Residual plot\n(pattern here = unmodeled structure)')
axes[1].grid(True, alpha=0.3)

# Plot 3: Actual vs Predicted (should hug the diagonal)
axes[2].scatter(y_test, y_pred, alpha=0.5, s=20)
mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
axes[2].plot([mn, mx], [mn, mx], 'r--', linewidth=1)
axes[2].set_xlabel('Actual'); axes[2].set_ylabel('Predicted')
axes[2].set_title(f'Actual vs Predicted\n(R² = {r2:.3f})')
axes[2].grid(True, alpha=0.3)

plt.suptitle('Linear Regression — Model Diagnostics', fontsize=13)
plt.tight_layout(); plt.show()

# ── 8. Interpret the model ────────────────────────────────────────────────

coef = best_model.named_steps['model'].coef_
feature_names = X.columns if hasattr(X, 'columns') else [f'x{i}' for i in range(len(coef))]

importance_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coef,
    'abs_importance': np.abs(coef)
}).sort_values('abs_importance', ascending=False)

print("\nFeature coefficients (after scaling — comparable in magnitude):")
print(importance_df.to_string(index=False))
print("\nInterpretation: a coefficient of 2.5 means increasing this feature")
print("by 1 standard deviation changes the prediction by 2.5 units.")
```

---

## 14.  Resources — Organized by Learning Goal

### To understand the intuition
| Resource | What to read/watch | What you'll get |
|---|---|---|
| Andrew Ng — ML Specialization (Coursera) | Week 1–2: Regression | The clearest visual derivation of gradient descent |
| 3Blue1Brown — Essence of Calculus | "What is a derivative?" | Derivatives as slopes — essential pre-math intuition |
| Distill.pub — "A Visual Explanation of Ridge Regression" | Full article | Why regularization works, beautifully illustrated |
| Jay Alammar — Visualizing ML | Linear Regression post | Diagram-heavy walkthrough |

### To understand the math
| Resource | What to read | What you'll get |
|---|---|---|
| Stanford CS229 Notes | Note 1: Supervised Learning | Full derivations — every step shown |
| *Pattern Recognition and ML* (Bishop) | Ch. 3: Linear Models | Probabilistic derivation, Bayesian treatment |
| *ESL* (Hastie, Tibshirani, Friedman) | Ch. 3: Linear Methods | Statistical theory, regularization geometry |
| *Mathematics for ML* (Deisenroth) | Ch. 9: Linear Regression | Clean unified treatment of MLE and MAP |

### To understand practical implementation
| Resource | What to read | What you'll get |
|---|---|---|
| *Hands-On ML* (Géron) | Ch. 4: Training Models | Best sklearn + visualization guide |
| Sklearn docs | LinearRegression, Ridge, Lasso | Full API with examples |
| Kaggle — House Prices competition | Any top notebook | Real feature engineering for linear models |

### To go deeper (research level)
| Paper | What you'll get |
|---|---|
| Tibshirani (1996) — "Regression Shrinkage and Selection via the Lasso" | Original Lasso paper — the geometry of sparse solutions |
| Hoerl & Kennard (1970) — "Ridge Regression" | Original Ridge paper — motivation from multicollinearity |
| Papers With Code — Linear Regression | State-of-the-art benchmarks across domains |

---

*← [Math Foundation](../1_Math_Foundation/) | [Table of Contents](../../README.md) | [Logistic Regression →](./logistic_regression.md)*

---

> **My notes** *(add your own "aha moments" here as you learn)*
>
> -
