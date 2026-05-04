#  ML Algorithms Cheat Sheet

> Quick revision reference for all major ML algorithms.

---

##  Supervised Learning

### Regression

| Algorithm | Type | Key Params | Pros | Cons |
|---|---|---|---|---|
| Linear Regression | Parametric | fit_intercept | Fast, interpretable | Assumes linearity |
| Ridge (L2) | Parametric | alpha | Handles multicollinearity | Doesn't zero weights |
| Lasso (L1) | Parametric | alpha | Feature selection | Unstable with correlated features |
| Elastic Net | Parametric | alpha, l1_ratio | Best of L1+L2 | Two hyperparams |
| Decision Tree Regressor | Non-parametric | max_depth | No assumptions | Overfits |
| Random Forest Regressor | Ensemble | n_estimators, max_depth | Robust, accurate | Slow, less interpretable |
| XGBoost Regressor | Ensemble | n_estimators, lr, depth | State of art | Needs tuning |
| SVR | Kernel-based | C, epsilon, kernel | Robust to outliers | Slow on large data |

### Classification

| Algorithm | Type | Key Params | Pros | Cons |
|---|---|---|---|---|
| Logistic Regression | Parametric | C, penalty | Fast, interpretable, probabilistic | Assumes linearity |
| KNN | Instance-based | n_neighbors, metric | Simple, no training | Slow prediction, scale-sensitive |
| Naive Bayes | Probabilistic | var_smoothing | Very fast, works well on text | Strong independence assumption |
| Decision Tree | Non-parametric | max_depth, criterion | Interpretable, handles mixed types | Unstable, overfits |
| Random Forest | Ensemble | n_estimators | Robust, feature importance | Less interpretable |
| SVM | Margin-based | C, kernel, gamma | Works in high-dim, effective | Slow, hard to interpret |
| XGBoost | Ensemble/Boosting | learning_rate, n_estimators, max_depth | High accuracy, regularization | Many hyperparams |
| LightGBM | Ensemble/Boosting | num_leaves, learning_rate | Faster than XGBoost on large data | Needs tuning |
| CatBoost | Ensemble/Boosting | iterations, depth | Handles categoricals natively | Slower training |
| Neural Network | Deep learning | layers, lr, batch_size | Universal approximator | Black box, needs lots of data |

---

##  Unsupervised Learning

### Clustering

| Algorithm | Type | Key Params | Pros | Cons |
|---|---|---|---|---|
| K-Means | Centroid | n_clusters, init | Fast, scalable | Assumes spherical clusters, needs k |
| K-Means++ | Centroid | n_clusters | Better initialization | Still needs k |
| DBSCAN | Density | eps, min_samples | Arbitrary shapes, finds outliers | Sensitive to eps |
| Hierarchical | Linkage | n_clusters, linkage | Dendrogram, no k needed | Slow O(n²) |
| Gaussian Mixture | Probabilistic | n_components | Soft assignments, flexible | Needs EM initialization |
| Mean Shift | Density | bandwidth | No k needed | Slow |

### Dimensionality Reduction

| Algorithm | Linear? | Key Params | Use Case |
|---|---|---|---|
| PCA | ✅ | n_components | Feature compression, visualization |
| LDA | ✅ | n_components | Supervised reduction (maximize class separation) |
| t-SNE | ❌ | perplexity | 2D/3D visualization only |
| UMAP | ❌ | n_neighbors, min_dist | Faster than t-SNE, preserves global structure |
| Autoencoder | ❌ | encoder layers | Non-linear compression |

---

##  Ensemble Methods Summary

```
Bagging (Parallel)          Boosting (Sequential)        Stacking
─────────────────           ──────────────────────        ──────────────────
  Training data               Training data               Base Models
  [Bootstrap 1] [Bootstrap 2]   │                         [Model 1] [Model 2]
       │              │          ▼                              │        │
  [Tree 1]      [Tree 2]    [Weak Learner 1]                   ▼        ▼
       │              │          │ ← residuals            [Predictions] [Predictions]
       └──── avg ─────┘     [Weak Learner 2]                    │
             │                   │ ← residuals            [Meta-Learner]
         Final Pred         [Final Ensemble]                    │
                                                           Final Prediction
```

---

##  Algorithm Selection Guide

```
Start here: What is your task?
│
├── Regression (continuous output)
│   ├── Linear relationship → Linear/Ridge/Lasso
│   ├── Non-linear, tabular → XGBoost/Random Forest
│   └── Image/text → Neural Networks
│
├── Classification (discrete output)
│   ├── Text data → Naive Bayes, Logistic, BERT
│   ├── Imbalanced data → XGBoost + class_weight
│   ├── Small dataset → SVM, Logistic Regression
│   ├── Large tabular → XGBoost, LightGBM
│   └── Image → CNN
│
├── Clustering (no labels)
│   ├── Know k → K-Means
│   ├── Arbitrary shapes → DBSCAN
│   └── Hierarchy needed → Hierarchical
│
└── Dimensionality Reduction
    ├── Visualization → t-SNE or UMAP
    ├── Feature reduction (linear) → PCA
    └── Feature reduction (supervised) → LDA
```

---

##  Loss Functions Reference

| Loss | Use | Formula |
|---|---|---|
| MSE | Regression | (1/n) Σ(yᵢ - ŷᵢ)² |
| MAE | Regression (robust) | (1/n) Σ\|yᵢ - ŷᵢ\| |
| RMSE | Regression (interpretable) | √MSE |
| Huber | Regression (outliers) | MSE if small, MAE if large |
| Binary Cross-Entropy | Binary classification | -[y log(ŷ) + (1-y)log(1-ŷ)] |
| Categorical Cross-Entropy | Multiclass | -Σ yᵢ log(ŷᵢ) |
| Hinge Loss | SVM | max(0, 1 - y·ŷ) |
| KL Divergence | Distribution comparison | Σ p(x) log(p(x)/q(x)) |

---

##  Key Metrics Quick Reference

### Classification
```
Accuracy  = (TP + TN) / Total            ← Don't use on imbalanced data
Precision = TP / (TP + FP)               ← When FP is costly
Recall    = TP / (TP + FN)               ← When FN is costly (cancer detection)
F1        = 2 × (P × R) / (P + R)       ← Balanced metric
AUC-ROC   = Area under TPR vs FPR curve  ← Threshold-independent
```

### Regression
```
MAE   = mean|y - ŷ|                 ← Interpretable, robust
MSE   = mean(y - ŷ)²                ← Penalizes large errors
RMSE  = √MSE                        ← Same unit as target
R²    = 1 - (SS_res / SS_tot)       ← 1 = perfect, 0 = baseline, <0 = worse than mean
MAPE  = mean(|y-ŷ|/y) × 100%       ← Percentage error
```

---

##  Regularization Cheat Sheet

```
No regularization:   Loss = L(y, ŷ)
L1 (Lasso):          Loss = L(y, ŷ) + λ Σ|wᵢ|       → Sparse weights
L2 (Ridge):          Loss = L(y, ŷ) + λ Σwᵢ²         → Small weights
Elastic Net:         Loss = L(y, ŷ) + λ₁Σ|wᵢ| + λ₂Σwᵢ²

Dropout:             Randomly zero out neurons during training
Early Stopping:      Stop training when validation loss starts increasing
Data Augmentation:   Expand training set artificially
```

---

##  Hyperparameter Tuning Methods

| Method | Description | Best For |
|---|---|---|
| Grid Search | Exhaustive search over param grid | Small param space |
| Random Search | Random sampling from param distributions | Large param space |
| Bayesian Optimization | Builds surrogate model; intelligently samples | Expensive models |
| Halving Search | Eliminates poor configs with more data | Fast screening |
| Optuna / Hyperopt | Modern Bayesian opt frameworks | Production use |

---

##  Validation Strategies

```
Simple Split:         [────── Train ──────][─ Val ─][─ Test ─]

k-Fold CV:            Fold 1: [Val][Tr][Tr][Tr][Tr]
                      Fold 2: [Tr][Val][Tr][Tr][Tr]
                      ...
                      Fold k: [Tr][Tr][Tr][Tr][Val]

Stratified k-Fold:   Same as k-Fold but each fold preserves class ratios

Time Series Split:    Train [1..t] → Val [t+1..t+n]  (never use future data for past)
                      Train [1..t+n] → Val [t+n+1..]
```

---

*Part of Complete ML Notes Repository | Quick revision before interviews*
