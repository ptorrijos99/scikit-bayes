.. title:: User guide : contents

.. _user_guide:

==========
User Guide
==========

This guide provides comprehensive documentation for scikit-bayes estimators,
including theoretical background and practical usage examples.

.. contents:: Table of Contents
   :local:
   :depth: 2

Introduction
============

What are Bayesian Network Classifiers?
--------------------------------------

Bayesian Network Classifiers (BNCs) are probabilistic classifiers based on 
Bayes' theorem. The simplest and most famous is **Naive Bayes**, which assumes
all features are conditionally independent given the class:

.. math::

    P(y|\mathbf{x}) \propto P(y) \prod_{i=1}^{n} P(x_i|y)

While this assumption rarely holds in practice, Naive Bayes is surprisingly
effective and computationally efficient. However, when feature dependencies
are strong (e.g., the XOR problem), it fails.

Why scikit-bayes?
-----------------

scikit-learn provides excellent Naive Bayes implementations, but has limitations:

1. **No native mixed data support**: You cannot directly combine Gaussian, 
   Categorical, and Bernoulli features in one model.

2. **No dependency modeling**: No implementations of AODE, A2DE, or other 
   n-dependence estimators that relax the independence assumption.

3. **No hybrid models**: No discriminatively-trained Bayesian classifiers
   like ALR (Accelerated Logistic Regression).

scikit-bayes fills these gaps with **fully scikit-learn compatible** estimators.


.. _mixed_naive_bayes:

MixedNB: Mixed Data Naive Bayes
===============================

:class:`skbayes.MixedNB` handles datasets with heterogeneous feature types
by internally combining scikit-learn's specialized Naive Bayes estimators.

The Problem
-----------

Consider a dataset with:

* **Feature 0**: Age (continuous) → Gaussian
* **Feature 1**: Gender (binary) → Bernoulli  
* **Feature 2**: Education Level (0, 1, 2, 3) → Categorical

With sklearn, you'd need to:

1. Split features by type
2. Fit separate NB classifiers
3. Combine probabilities manually

MixedNB does this automatically.

Feature Type Detection
----------------------

MixedNB auto-detects feature types during ``fit()``:

* **Gaussian**: Float features with non-integer values
* **Bernoulli**: Features with exactly 2 unique values
* **Categorical**: Integer features with >2 unique values

You can also specify types manually:

.. code-block:: python

    from skbayes import MixedNB
    
    # Force features 2 and 3 to be categorical
    clf = MixedNB(categorical_features=[2, 3])
    
    # Force feature 1 to be Bernoulli
    clf = MixedNB(bernoulli_features=[1])

Usage Example
-------------

.. code-block:: python

    import numpy as np
    from skbayes import MixedNB

    # Features: [Gaussian, Bernoulli, Categorical]
    X = np.array([
        [1.5, 0, 0],
        [2.3, 1, 1],
        [0.8, 1, 2],
        [1.1, 0, 0],
        [3.2, 1, 1],
        [-0.5, 0, 2]
    ])
    y = np.array([0, 1, 1, 0, 1, 0])

    clf = MixedNB(alpha=1.0)
    clf.fit(X, y)

    # Inspect detected types
    print(clf.feature_types_)
    # {'gaussian': [0], 'bernoulli': [1], 'categorical': [2]}

    # Predict
    print(clf.predict([[1.0, 1, 1]]))  # [1]
    print(clf.predict_proba([[1.0, 1, 1]]))  # [[0.xx, 0.xx]]

Parameters
----------

* ``alpha``: Smoothing parameter (Laplace smoothing) for Categorical/Bernoulli. Default: 1.0
* ``var_smoothing``: Variance smoothing for Gaussian features. Default: 1e-9
* ``categorical_features``: List of indices to treat as categorical
* ``bernoulli_features``: List of indices to treat as Bernoulli


AnDE Family: Relaxing Independence
==================================

The **Averaged n-Dependence Estimators (AnDE)** family relaxes the independence
assumption by conditioning on "super-parent" features.

The Independence Problem
------------------------

Consider the XOR problem:

.. list-table::
   :header-rows: 1

   * - X1
     - X2
     - Y
   * - -1
     - -1
     - 0
   * - -1
     - +1
     - 1
   * - +1
     - -1
     - 1
   * - +1
     - +1
     - 0

Looking at X1 alone: P(X1|Y=0) = P(X1|Y=1) (symmetric distributions).
Looking at X2 alone: P(X2|Y=0) = P(X2|Y=1) (symmetric distributions).

**Naive Bayes cannot learn this.** It achieves ~50% accuracy (random guessing).

AnDE solves this by modeling **P(X2 | Y, X1)** instead of just P(X2 | Y).

The Super-Parent Strategy
-------------------------

An SPnDE (Super-Parent n-Dependence Estimator) conditions all child features
on the class **and** n parent features:

.. math::

    P(y, \mathbf{x}) = P(Y^*) \prod_{i \in children} P(x_i | Y^*)

Where :math:`Y^* = (y, x_{p1}, x_{p2}, ..., x_{pn})` is the "augmented super-class".

AnDE averages over all possible parent combinations.

AnDE (Arithmetic Mean)
----------------------

:class:`skbayes.AnDE` is the standard generative model described by Webb et al. [1]_.

.. math::

    P(y|\mathbf{x}) \propto \sum_{m} P_m(y, \mathbf{x})

**Key Parameters:**

* ``n_dependence``: Order of dependence
  
  - n=0: Equivalent to Naive Bayes (MixedNB)
  - n=1: AODE (Averaged One-Dependence Estimators)
  - n=2: A2DE (common choice for higher accuracy)

* ``n_bins``: Discretization bins for continuous super-parents
* ``strategy``: Discretization strategy ('uniform', 'quantile', 'kmeans')

**Example:**

.. code-block:: python

    from skbayes import AnDE

    # AODE (n=1)
    clf = AnDE(n_dependence=1, n_bins=5)
    clf.fit(X, y)
    
    # A2DE (n=2) - higher accuracy, more computation
    clf = AnDE(n_dependence=2, n_bins=5)

AnJE (Geometric Mean)
---------------------

:class:`skbayes.AnJE` aggregates using the geometric mean (product of probabilities):

.. math::

    P(y|\mathbf{x}) \propto \prod_{m} P_m(y, \mathbf{x})

This is equivalent to summing log-probabilities and serves as the basis for
the convex ALR optimization.

**Usage is identical to AnDE.**

ALR: Accelerated Logistic Regression
------------------------------------

:class:`skbayes.ALR` is a **hybrid generative-discriminative** classifier [2]_.

It starts with the AnJE generative model and learns discriminative weights
to optimize classification performance:

.. math::

    P(y|\mathbf{x}) \propto \exp\left(\sum_{m} w_m \cdot \log P_m(y, \mathbf{x})\right)

**Weight Granularity Levels:**

ALR supports 4 levels of parameter granularity:

.. list-table::
   :header-rows: 1
   :widths: 10 30 30 30

   * - Level
     - Description
     - # Parameters
     - Best For
   * - 1
     - Per Model
     - M
     - Small datasets
   * - 2
     - Per Parent Value
     - M × V
     - Large datasets
   * - 3
     - Per Class
     - M × C
     - Multi-class
   * - 4
     - Per Value × Class
     - M × V × C
     - Very large datasets

Where M = number of models, V = parent value combinations, C = classes.

**Example:**

.. code-block:: python

    from skbayes import ALR

    # Level 1: Simple, low variance
    clf = ALR(n_dependence=1, weight_level=1, l2_reg=1e-3)
    
    # Level 3: Per-class weights (good for multi-class)
    clf = ALR(n_dependence=1, weight_level=3, l2_reg=1e-4)
    
    clf.fit(X, y)

WeightedAnDE
------------

:class:`skbayes.WeightedAnDE` applies discriminative weighting to the standard
AnDE (arithmetic mean) model. Unlike ALR, the optimization is **non-convex**.

.. code-block:: python

    from skbayes import WeightedAnDE
    
    clf = WeightedAnDE(n_dependence=1, weight_level=1)
    clf.fit(X, y)


Parameter Tuning Guide
======================

Choosing n_dependence
---------------------

* **n=1 (AODE)**: Good default. Captures pairwise interactions.
* **n=2 (A2DE)**: Better accuracy, but O(n²) models. Use for <50 features.
* **n≥3**: Rarely needed. Computational cost grows combinatorially.

Discretization Strategy
-----------------------

For continuous super-parents:

* **'quantile'** (default): Equal-frequency bins. Robust to outliers.
* **'uniform'**: Equal-width bins. Good for uniform distributions.
* **'kmeans'**: Data-driven bins. Best for multi-modal distributions.

``n_bins`` typically 3-10. More bins = more precision but fewer samples per bin.

Regularization in Hybrid Models
-------------------------------

``l2_reg`` controls regularization strength:

* **Small datasets**: Use higher values (1e-2 to 1e-1) to prevent overfitting
* **Large datasets**: Use lower values (1e-4 to 1e-3) for more flexibility

Computational Considerations
----------------------------

* Use ``n_jobs=-1`` to parallelize SPODE fitting
* Higher ``weight_level`` increases optimization time exponentially
* A2DE with n_features=50 creates ~1,225 sub-models


References
==========

.. [1] Webb, G. I., Boughton, J., & Wang, Z. (2005). Not so naive Bayes: 
       Aggregating one-dependence estimators. Machine Learning, 58(1), 5-24.

.. [2] Flores, M. J., Gámez, J. A., Martínez, A. M., & Puerta, J. M. (2009).
       GAODE and HAODE: Two proposals based on AODE to deal with continuous 
       variables. ICML '09, 313-320.

.. [3] Zaidi, N. A., Webb, G. I., Carman, M. J., & Petitjean, F. (2017). 
       Efficient parameter learning of Bayesian network classifiers. 
       Machine Learning, 106(9-10), 1289-1329.
