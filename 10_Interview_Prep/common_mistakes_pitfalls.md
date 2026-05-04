#  Common ML Mistakes & Interview Pitfalls

> These are the things interviewers love to test and candidates love to get wrong.
> Read these carefully — they're often the difference between pass and fail.

---

##  Data & Preprocessing Pitfalls

### 1. Fitting preprocessing on the full dataset (Data Leakage)
```python
# ❌ WRONG — scaler sees test data during fit
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # fit on ALL data
X_train, X_test = train_test_split(X_scaled, ...)

# ✅ CORRECT — fit only on train, transform both
X_train, X_test = train_test_split(X, ...)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # only transform, no fit
```

**Why it matters:** Test set statistics leak into the model, giving falsely optimistic performance.

---

### 2. Reporting accuracy on imbalanced datasets
```python
# ❌ WRONG — 95% accuracy might mean the model predicts majority class always
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))  # 95%... but is it meaningful?

# ✅ CORRECT — use F1, AUC-ROC, confusion matrix
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

**Rule of thumb:** If class distribution is >80:20, accuracy is misleading.

---

### 3. Ignoring the baseline
> Never present a model without comparing it to a simple baseline.

- **Classification baseline:** Majority class classifier
- **Regression baseline:** Predict mean
- **Time series baseline:** Yesterday's value / last known value

If your model doesn't beat the baseline, something is wrong.

---

### 4. Not checking for missing values before modeling
```python
# Always do this before fitting
print(df.isnull().sum())
print(df.isnull().mean() * 100)  # percentage missing per column
```

Missing values cause silent errors or NaN predictions.

---

### 5. Not checking data types
```python
print(df.dtypes)
print(df.describe())
```
Categorical columns stored as `int` (like zip codes) will be treated as numeric features.

---

##  Model Building Pitfalls

### 6. Tuning hyperparameters on the test set
```python
# ❌ WRONG
for C in [0.1, 1, 10]:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # using test set to pick C!

# ✅ CORRECT — use a validation set or cross-validation
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(model, {'C': [0.1, 1, 10]}, cv=5)
gs.fit(X_train, y_train)  # cv is done entirely within train
gs.score(X_test, y_test)  # test set used ONLY once at the end
```

---

### 7. Assuming more features always help
Adding irrelevant features can:
- Increase variance
- Hurt interpretability
- Slow down training
- Reduce performance in distance-based models (curse of dimensionality)

Always use feature selection or regularization when adding many features.

---

### 8. Confusing model parameters with hyperparameters
- **Parameters:** Learned from data (weights, biases) — set by the model during training
- **Hyperparameters:** Set before training (learning rate, n_estimators, depth) — you control these

This is a common interview question!

---

### 9. Using the wrong loss function
| Task | Wrong | Right |
|---|---|---|
| Binary classification | MSE | Binary cross-entropy |
| Multiclass | Binary cross-entropy | Categorical cross-entropy |
| Imbalanced classes | Plain cross-entropy | Weighted cross-entropy / Focal loss |
| Regression with outliers | MSE | MAE or Huber |

---

### 10. Not doing cross-validation on small datasets
```python
# ❌ WRONG — single split on small dataset is unreliable
X_train, X_test = train_test_split(X, test_size=0.2)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)  # could be very lucky or unlucky

# ✅ CORRECT — use k-fold CV
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=10)
print(f"Mean: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

##  Deep Learning Pitfalls

### 11. Not normalizing inputs for neural networks
Neural networks are extremely sensitive to input scale. Always normalize:
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
```

### 12. Using sigmoid/tanh in hidden layers
- Sigmoid and tanh cause vanishing gradients in deep networks
- **Default:** Use ReLU or LeakyReLU for hidden layers
- Only use sigmoid for binary output, softmax for multiclass output

### 13. Not using appropriate weight initialization
- Zero initialization: all neurons learn the same thing (symmetry breaking problem)
- Use **He initialization** for ReLU, **Xavier** for sigmoid/tanh
```python
# PyTorch
nn.Linear(in, out)  # default is Kaiming/He — good for ReLU
```

### 14. Forgetting to set model to eval mode during inference
```python
# PyTorch
model.eval()  # turns off dropout, sets BN to use running stats
with torch.no_grad():
    predictions = model(X_test)
```

### 15. Not monitoring validation loss during training
Train loss going down doesn't mean the model is improving. Always plot both:
```python
# Track both
history = model.fit(X_train, y_train, validation_data=(X_val, y_val))
```
If val loss increases while train loss decreases → overfitting.

---

##  Interview-Specific Pitfalls

### 16. Saying "higher accuracy is always better"
❌ Wrong. Accuracy is misleading on imbalanced datasets.

Correct response: *"It depends on the class balance and the cost of different error types. For imbalanced data, I'd use F1-score or AUC-ROC."*

### 17. Not knowing the difference between parameters and hyperparameters
This is a very basic question. Know it cold.

### 18. Saying regularization "prevents overfitting" without explaining how
Explain: *"Regularization adds a penalty term to the loss function that discourages large weights, making the model simpler and less likely to overfit the training data."*

### 19. Saying PCA is for removing "unimportant" features
PCA doesn't remove features based on importance — it creates new components that are linear combinations of all features, ranked by variance explained. It's unsupervised and doesn't know the target.

### 20. Not knowing when NOT to use a specific algorithm
Every algorithm interview question should end with: *"...but I wouldn't use this when [limitation]."*

Example: *"KNN is simple and intuitive, but I wouldn't use it on large datasets because prediction is O(n) and it's very sensitive to feature scaling and irrelevant features."*

---

##  Conceptual Traps

### 21. Confusing correlation and causation
Ice cream sales correlate with drowning deaths. Not because ice cream causes drowning, but because both increase in summer (the confounder).

In interviews: Always mention confounders when discussing correlation.

### 22. Thinking cross-validation gives you more data
Cross-validation is a way to get a better **estimate** of model performance. It does NOT create new data. The model is still trained on the same samples.

### 23. Thinking ensemble methods always work
- Bagging reduces variance — won't help if model has high bias
- Boosting reduces bias — can overfit if not regularized
- Stacking is only as good as the diversity of base learners

### 24. Misunderstanding the bias-variance tradeoff
- Bias and variance are properties of the **learning algorithm**, not a specific trained model
- They describe expected behavior across many training sets
- In practice, we don't compute them directly — we use learning curves to diagnose

### 25. Not understanding what "generalization" means
Generalization = performance on **unseen** data. A model generalizes well if its train and test performance are similar and both are good.

---

##  Golden Rules to Remember

```
1. Always split BEFORE any preprocessing
2. Never use test set during training or tuning
3. Accuracy is misleading on imbalanced data
4. Compare against a baseline first
5. Check for data leakage — especially in time series
6. More data helps variance, not bias
7. Understand WHY an algorithm works, not just WHAT it does
8. Every model choice is a tradeoff — know yours
9. Always state assumptions when explaining algorithms
10. In interviews: say "it depends" and then explain the tradeoff
```

---

*Part of Complete ML Notes Repository | Know these cold before your interview*
