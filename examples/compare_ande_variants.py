"""
====================================================================
Benchmark: Generative vs. Hybrid AnDE on Mixed Data
====================================================================

This example compares the performance and computational cost of the four
variants of the AnDE family implemented in scikit-bayes:

1.  **AnDE**: Generative, Arithmetic Mean (The classic AODE/A2DE).
2.  **AnJE**: Generative, Geometric Mean (The base for ALR).
3.  **ALR**: Hybrid, Geometric Mean (Convex Optimization).
4.  **WeightedAnDE**: Hybrid, Arithmetic Mean (Non-Convex Optimization).

**Experimental Setup:**
-   **Data:** Synthetic dataset with 10 features (5 Continuous, 5 Categorical).
-   **Task:** Binary classification with noise (class overlap) to make
    probability estimation challenging.
-   **Metrics:**
    -   **Training Time:** Measures the cost of the optimization phase.
    -   **Log Loss:** The primary metric for probability calibration (ALR optimizes this).
    -   **Accuracy:** The standard classification metric.
"""

# Author: The scikit-bayes Developers
# SPDX-License-Identifier: BSD-3-Clause

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import KBinsDiscretizer

from skbayes.ande import AnDE, AnJE, ALR, WeightedAnDE

# --- 1. Generate Complex Mixed Dataset ---
print("Generating dataset...")
n_samples = 20000
n_features = 10
# Generate standard continuous data
X, y = make_classification(
    n_samples=n_samples, 
    n_features=n_features,
    n_informative=6, 
    n_redundant=2, 
    n_classes=2,
    flip_y=0.1, # Add noise (10% labels flipped)
    random_state=42
)

# Make it MIXED: Force the first 5 features to be Categorical (Integers 0..4)
est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
X[:, :5] = est.fit_transform(X[:, :5]).astype(int)

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 2. Define Models to Benchmark ---
common_params = {'n_dependence': 1, 'n_bins': 5}
hybrid_params = {'max_iter': 100, 'l2_reg': 1e-3}

models = [
    ("AnDE (Gen Aritm)", AnDE(**common_params)),
    ("AnJE (Gen Geom)", AnJE(**common_params)),
    ("WeightedAnDE (Hyb Aritm)", WeightedAnDE(**common_params, **hybrid_params)),
    ("ALR (Hyb Geom)", ALR(**common_params, **hybrid_params)),
]

# --- 3. Run Benchmark ---
results = []

for name, clf in models:
    print(f"Training {name}...")
    
    # 1. Training Time
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # 2. Inference
    y_pred_test = clf.predict(X_test)
    y_prob_test = clf.predict_proba(X_test)
    
    # 3. Metrics
    acc = accuracy_score(y_test, y_pred_test)
    ll = log_loss(y_test, y_prob_test)
    
    # Store
    results.append({
        "Model": name,
        "Time (s)": train_time,
        "Accuracy": acc,
        "Log Loss": ll
    })

# --- 4. Display Results (Console) ---
print("\nBenchmark Results:")
print(f"{'Model':<30} | {'Time (s)':<10} | {'Accuracy':<10} | {'Log Loss':<10}")
print("-" * 70)
for r in results:
    print(f"{r['Model']:<30} | {r['Time (s)']:<10.4f} | {r['Accuracy']:<10.4f} | {r['Log Loss']:<10.4f}")

# --- 5. Visualization (Improved) ---
model_names = [r['Model'] for r in results]
times = [r['Time (s)'] for r in results]
accs = [r['Accuracy'] for r in results]
lls = [r['Log Loss'] for r in results]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot Time
bars_time = axes[0].bar(model_names, times, color=['skyblue', 'skyblue', 'orange', 'orange'])
axes[0].set_title("Training Time (Lower is Better)")
axes[0].set_ylabel("Seconds")
axes[0].bar_label(bars_time, fmt='%.3f')
axes[0].tick_params(axis='x', rotation=20)

# Plot Accuracy (Dynamic Zoom)
bars_acc = axes[1].bar(model_names, accs, color='lightgreen')
axes[1].set_title("Test Accuracy (Higher is Better)")
# Dynamic limits: Focus on the variation
min_acc = min(accs)
max_acc = max(accs)
margin = (max_acc - min_acc) * 2 if max_acc != min_acc else 0.01
axes[1].set_ylim(max(0, min_acc - margin), min(1, max_acc + margin))
axes[1].bar_label(bars_acc, fmt='%.4f')
axes[1].tick_params(axis='x', rotation=20)

# Plot Log Loss
bars_ll = axes[2].bar(model_names, lls, color='salmon')
axes[2].set_title("Test Log Loss (Lower is Better)")
axes[2].set_ylabel("NLL")
axes[2].bar_label(bars_ll, fmt='%.3f')
axes[2].tick_params(axis='x', rotation=20)

plt.suptitle(f"AnDE Family Benchmark (n=1) on Mixed Data ({n_samples} samples)", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.85, bottom=0.2) # Adjust bottom for rotated labels
plt.show()