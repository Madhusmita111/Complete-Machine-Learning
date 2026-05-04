#  ML Concepts: Side-by-Side Comparisons

> One of the most tested areas in interviews. Master the "X vs Y" questions.

---

##  Algorithms

### Linear Regression vs Logistic Regression

| | Linear Regression | Logistic Regression |
|---|---|---|
| Output | Continuous value | Probability (0 to 1) |
| Output layer | Identity | Sigmoid |
| Loss | MSE | Binary Cross-Entropy |
| Use case | Predict house prices | Classify spam/not-spam |
| Decision boundary | Not applicable | Linear |
| Assumptions | Residuals ~ Normal | Log-odds are linear in X |

---

### L1 (Lasso) vs L2 (Ridge) Regularization

| | L1 (Lasso) | L2 (Ridge) |
|---|---|---|
| Penalty | Σ\|wᵢ\| | Σwᵢ² |
| Gradient | ±λ (constant) | 2λwᵢ (proportional to weight) |
| Weight behavior | Drives weights to exact 0 | Shrinks weights, never 0 |
| Feature selection | ✅ Yes | ❌ No |
| When features are correlated | Picks one, ignores rest | Distributes weight across all |
| Geometry | Diamond constraint region | Circular constraint region |
| Solution | Non-smooth, no closed form | Smooth, has closed form |

---

### Random Forest vs XGBoost vs LightGBM

| | Random Forest | XGBoost | LightGBM |
|---|---|---|---|
| Type | Bagging | Boosting | Boosting |
| Trees | Parallel | Sequential | Sequential |
| Bias | Low | Lower | Lower |
| Variance | Low | Low | Low |
| Speed | Moderate | Slower | Fastest |
| Memory | High | Moderate | Low |
| Overfitting | Resistant | More prone | More prone |
| Handles missing | ❌ | ✅ | ✅ |
| Categoricals | Needs encoding | Needs encoding | ✅ Native (partially) |
| Tuning | Easier | Harder | Harder |

---

### SVM vs Logistic Regression

| | SVM | Logistic Regression |
|---|---|---|
| Objective | Maximize margin | Maximize likelihood |
| Output | Class label | Probability |
| Kernel support | ✅ (non-linear) | ❌ (linear only without tricks) |
| Outlier sensitivity | Less sensitive (only support vectors matter) | Sensitive |
| Dataset size | Best on small-medium | Scales better |
| Interpretability | Low | High |
| Training time | Slow on large data | Fast |

---

### Decision Tree vs Random Forest

| | Decision Tree | Random Forest |
|---|---|---|
| Architecture | Single tree | Ensemble of trees |
| Bias | Low | Low |
| Variance | High | Low |
| Overfitting | Prone | Resistant |
| Interpretability | High | Low |
| Speed | Fast | Slower |
| Feature selection | Full features per split | Random subset per split |

---

### KNN vs K-Means

| | KNN | K-Means |
|---|---|---|
| Type | Supervised | Unsupervised |
| K means | # neighbors | # clusters |
| Training | None (lazy) | Iterative |
| Prediction | Distance to neighbors | Distance to centroids |
| Use | Classification/Regression | Clustering |

---

### Bagging vs Boosting

| | Bagging | Boosting |
|---|---|---|
| Training | Parallel | Sequential |
| Sample weights | Equal (bootstrap) | Weighted by error |
| Goal | Reduce variance | Reduce bias |
| Overfitting | Less prone | More prone |
| Example | Random Forest | XGBoost, AdaBoost |

---

##  Evaluation Metrics

### Precision vs Recall vs F1

| | Precision | Recall | F1 |
|---|---|---|---|
| Formula | TP/(TP+FP) | TP/(TP+FN) | 2·P·R/(P+R) |
| Question | Of all predicted positives, how many are truly positive? | Of all actual positives, how many did we catch? | Balanced score |
| Use when | FP is costly (spam filter — missing important email is bad) | FN is costly (cancer detection — missing cancer is deadly) | Classes are imbalanced |
| Tradeoff | Increasing threshold → higher precision, lower recall | ↑ | Combined |

---

### AUC-ROC vs Precision-Recall Curve

| | AUC-ROC | PR Curve |
|---|---|---|
| X-axis | FPR (False Positive Rate) | Recall |
| Y-axis | TPR (Recall) | Precision |
| Threshold | Independent | Independent |
| Best for | Balanced datasets | Highly imbalanced datasets |
| Interpretation | Higher = better (1.0 = perfect, 0.5 = random) | Higher area = better |

---

### R² vs Adjusted R²

| | R² | Adjusted R² |
|---|---|---|
| Formula | 1 - SS_res/SS_tot | 1 - (1-R²)(n-1)/(n-p-1) |
| Behavior with more features | Always increases or stays same | Penalizes unnecessary features |
| Use | Quick look | Model comparison with different # features |

---

##  Deep Learning

### CNN vs RNN vs Transformer

| | CNN | RNN | Transformer |
|---|---|---|---|
| Input type | Grid (images) | Sequences | Any (via tokenization) |
| Memory | Local (kernels) | Hidden state | Full attention |
| Parallelizable | ✅ | ❌ | ✅ |
| Long-range deps | ❌ | Struggles (vanishing grad) | ✅ (attention) |
| Examples | ResNet, VGG | LSTM, GRU | BERT, GPT, T5 |

---

### LSTM vs GRU

| | LSTM | GRU |
|---|---|---|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| Cell state | Separate long-term memory | Merged with hidden state |
| Parameters | More | Fewer |
| Performance | Better on complex tasks | Often comparable, faster |
| Training | Slower | Faster |

---

### BERT vs GPT

| | BERT | GPT |
|---|---|---|
| Architecture | Encoder | Decoder |
| Training task | Masked LM (bidirectional) | Next token prediction (left-to-right) |
| Context | Sees full sequence | Only left context |
| Best for | NLU tasks (classification, NER, QA) | Text generation |
| Fine-tuning | Add classification head | Few-shot or fine-tune |

---

### Dropout vs Batch Normalization

| | Dropout | Batch Normalization |
|---|---|---|
| Purpose | Regularization | Stabilize training |
| How | Randomly zeroes neurons | Normalizes layer inputs |
| Applied during inference | ❌ (turned off) | ✅ (uses running stats) |
| Effect | Reduces overfitting | Speeds up training, acts as regularizer |

---

### Max Pooling vs Average Pooling

| | Max Pooling | Average Pooling |
|---|---|---|
| Operation | Takes max value in window | Takes average of window |
| Preserves | Sharp features, edges | Smooth, distributed features |
| Common use | CNNs (features, early layers) | CNNs (final spatial reduction) |

---

##  Data & Features

### Normalization vs Standardization

| | Normalization (Min-Max) | Standardization (Z-score) |
|---|---|---|
| Formula | (x - min) / (max - min) | (x - μ) / σ |
| Range | [0, 1] | Unbounded |
| Sensitive to outliers | ✅ | Less so |
| Use when | Bounded range needed, NNs | Normal-ish distribution |
| Example algorithms | KNN, Neural Networks | SVM, PCA, Logistic Regression |

---

### PCA vs LDA

| | PCA | LDA |
|---|---|---|
| Type | Unsupervised | Supervised |
| Goal | Maximize variance | Maximize class separability |
| Uses labels | ❌ | ✅ |
| Output components | n (up to features) | c-1 (c = # classes) |
| Use case | Compression, visualization | Feature extraction before classification |

---

### One-Hot Encoding vs Label Encoding vs Target Encoding

| | One-Hot | Label | Target |
|---|---|---|---|
| Output | Binary columns per category | Single integer column | Mean of target per category |
| Dimensionality | High (one col per category) | Low | Low |
| Implies order? | ❌ | ✅ (false) | ❌ |
| Risk | Dummy variable trap, sparse | Ordinal relationship implied | Data leakage risk |
| Good for | Nominal categories, LR, DL | Tree-based models only | Tree-based models (with care) |

---

##  Conceptual

### Generative vs Discriminative Models

| | Generative | Discriminative |
|---|---|---|
| Models | P(X,Y) or P(X\|Y) | P(Y\|X) directly |
| Can generate samples | ✅ | ❌ |
| Needs more data | ✅ | ❌ |
| Generally more accurate for classification | ❌ | ✅ |
| Examples | Naive Bayes, GMM, GAN | Logistic Regression, SVM, NNs |

---

### Parametric vs Non-Parametric Models

| | Parametric | Non-Parametric |
|---|---|---|
| # Parameters | Fixed | Grows with data |
| Storage | Low | High (stores data) |
| Training | Fast | N/A (lazy) |
| Prediction | Fast | Slow (search) |
| Assumptions | Strong | Few |
| Examples | Linear/Logistic Regression, Naive Bayes | KNN, Decision Trees, Kernel SVM |

---

### Online vs Batch Learning

| | Online Learning | Batch Learning |
|---|---|---|
| Training | On each new sample | On full dataset |
| Memory | Low | High |
| Adapts to new data | ✅ | ❌ (retrain needed) |
| Stability | Less stable | More stable |
| Examples | Streaming data, ad systems | Most offline ML workflows |

---

*Part of Complete ML Notes Repository | Master these comparisons for interviews*
