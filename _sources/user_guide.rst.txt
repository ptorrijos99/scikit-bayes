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

Bayesian Network Classifiers (BNCs) compute class probabilities using Bayes' theorem. The most common is Naive Bayes, which assumes features are conditionally independent given the class:

.. math::

    P(y|\mathbf{x}) \propto P(y) \prod_{i=1}^{n} P(x_i|y)

This assumption is rarely true, but Naive Bayes scales well and works as a baseline. When feature dependencies are strong (e.g., in a parity problem like XOR), it produces inaccurate probabilities.

Why scikit-bayes?
-----------------

While scikit-learn includes standard Naive Bayes classes, they require some manual workarounds for complex datasets:

1. **Heterogeneous data**: You cannot pass a matrix with Gaussian, Categorical, and Bernoulli features directly into a single estimator without using a `ColumnTransformer` to split them.
2. **Feature dependencies**: There are no out-of-the-box implementations of n-dependence estimators like AODE or A2DE.
3. **Hybrid optimization**: Scikit-learn does not implement discriminatively-weighted Bayesian classifiers like ALR, which use generative models as preconditioners.

`scikit-bayes` implements these architectures while passing standard scikit-learn estimator checks.


.. _mixed_naive_bayes:

MixedNB: Mixed Data Naive Bayes
===============================

:class:`skbn.MixedNB` handles datasets with heterogeneous feature types
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

    from skbn import MixedNB

    # Force features 2 and 3 to be categorical
    clf = MixedNB(categorical_features=[2, 3])

    # Force feature 1 to be Bernoulli
    clf = MixedNB(bernoulli_features=[1])

Usage Example
-------------

.. code-block:: python

    import numpy as np
    from skbn import MixedNB

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

:class:`skbn.AnDE` is the standard generative model described by Webb et al. [1]_.

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

    from skbn import AnDE

    # AODE (n=1)
    clf = AnDE(n_dependence=1, n_bins=5)
    clf.fit(X, y)

    # A2DE (n=2) - higher accuracy, more computation
    clf = AnDE(n_dependence=2, n_bins=5)

With ``mean="geometric"`` the components are pooled with the geometric mean,
:math:`P(y|\mathbf{x}) \propto \prod_{m} P_m(y, \mathbf{x})^{1/|M|}`. Every
component then counts the same, so a few components that give a class a
probability close to zero rule it out. This pool of SPnDEs is not the AnJE of
Zaidi et al. [4]_, which is described below.

WeightedAnDE
------------

:class:`skbn.WeightedAnDE` learns discriminative weights on top of AnDE by
maximising the conditional log-likelihood with L-BFGS-B. The weights start at,
and are regularized towards, the generative model (``w = 1``). Two parameters
define the model:

* ``mean``: how the components are pooled (``"arithmetic"`` or ``"geometric"``).
* ``weight_scope``: what each weight multiplies.

  - ``"component"``: a whole SPnDE. With the arithmetic mean the weights are
    mixture coefficients, :math:`\log \sum_m w_m P_m(y, \mathbf{x})`; with the
    geometric mean they scale the log-joints,
    :math:`\sum_m w_m \log P_m(y, \mathbf{x})`.
  - ``"factor"``: every factor of every SPnDE, as in the weighted CCBN of
    Zaidi et al. [3]_: :math:`P_m^w = \prod_f \theta_f^{w_f}`.

``weight_level`` sets the variables that index each weight:

.. list-table::
   :header-rows: 1
   :widths: 10 45 45

   * - Level
     - Component Weights
     - Factor Weights [3]_
   * - 1
     - Component
     - Factor
   * - 2
     - Component × parent value
     - Factor × value
   * - 3
     - Component × class
     - Factor × value × class
   * - 4
     - Component × class × parent value
     - Factor × value × class × parents

The factor levels are the nested parameterizations of Zaidi et al. [3]_
(Sec. 5.4), applied inside each SPnDE. Arithmetic component weights at level 3
are the per-SPODE, per-class weights of EODE-CLL [5]_. The other component
levels follow the same nesting at the level of the component; they are a
choice of this library, not of the literature.

Component weights are cheap (one weight per component and sample) but only
decide how much each SPnDE is trusted: they cannot correct a dependence
between children that every SPnDE shares, such as a duplicated attribute.
Factor weights can, at the cost of about ``n_features`` times more weights.
With the geometric mean, and with ``modular=True`` (each component fitted on
its own conditional log-likelihood), the objective is convex.

``average`` decides what the arithmetic mean pools: the weighted joints
:math:`P^w_m(y, \mathbf{x})`, as AnDE does (``"joint"``, the default), or the
components' posteriors :math:`P^w_m(y | \mathbf{x})` (``"posterior"``). The two
coincide for the geometric mean. Factor weights at level 4, fitted with
``modular=True`` and pooled with ``average="posterior"``, are the AnDE\
:sup:`w` of FedAnDE: one weighted CCBN per SPnDE, trained on its own data and
averaged, which is what lets each component be fitted across sites.

.. code-block:: python

    from skbn import WeightedAnDE

    # One weight per SPODE and class
    clf = WeightedAnDE(n_dependence=1, weight_level=3)

    # Weighted CCBN inside each SPODE
    clf = WeightedAnDE(n_dependence=1, weight_scope="factor", weight_level=3)

    # AnDE^w of FedAnDE
    clf = WeightedAnDE(
        n_dependence=1, weight_scope="factor", weight_level=4,
        modular=True, average="posterior",
    )
    clf.fit(X, y)

AnJE and ALR
------------

:class:`skbn.AnJE` and :class:`skbn.ALR` follow Zaidi et al. [4]_. Instead of
SPnDEs they use n-joins, the joint distributions :math:`P(x_\alpha | y)` of
every subset :math:`\alpha` of ``n_join`` attributes. AnJE pools them with
the geometric-mean weight of the paper:

.. math::

    P(y|\mathbf{x}) \propto P(y) \prod_{\alpha} P(x_\alpha | y)^{\frac{(n-1)!(a-n)!}{(a-1)!}}

ALR (ALR\ :sup:`n` in the paper) learns one weight per n-join, value and
class by maximising the conditional log-likelihood, a convex problem:

.. math::

    \log P(y|\mathbf{x}) = w_y \log P(y) + \sum_{\alpha} w_{y,\alpha,x_\alpha} \log P(x_\alpha | y) - \log Z(\mathbf{x})

A1JE is Naive Bayes, and AnJE uses the same statistics as A(n-1)DE, so
``AnJE(n_join=2)`` is comparable to ``AnDE(n_dependence=1)``. Both models are
defined on discrete attributes: continuous attributes are discretized into
``n_bins`` bins (the paper uses MDL discretization, available as
:class:`skbn.MDLPDiscretizer`). Probabilities are m-estimates with
``m = alpha``.

``weight_level=3`` (the default) is the model of the paper. Levels 1 (one
weight per n-join) and 2 (per n-join and value) tie the weights following the
nesting of [3]_. The other defaults reproduce the protocol of the paper's
experiments: weights start at ``w = 1``, there is no penalty, and L-BFGS-B
runs until the relative reduction of the objective falls below 1e-32 or for
10\ :sup:`4` iterations. The starting point is not the AnJE but the unaveraged
product of all n-joins, so on small samples the unpenalized model overfits.
``l2_reg`` adds the penalty towards ``w = 1`` that the paper suggests for
small datasets, and ``max_iter`` and ``tol`` stop the optimizer earlier.

.. code-block:: python

    from skbn import ALR, AnJE

    clf = AnJE(n_join=2)
    clf = ALR(n_join=2)  # as in Zaidi et al. (2016)
    clf.fit(X, y)


.. _tan:

TAN: Tree Augmented Naive Bayes
===============================

:class:`skbn.TAN` [6]_ relaxes the independence assumption in a different way
from AnDE: instead of averaging many models, it learns a single network in
which every attribute has the class and **at most one other attribute** as
parents.

.. math::

    P(y, \mathbf{x}) = P(y) \, P(x_r | y) \prod_{i \neq r} P(x_i | y, x_{\pi(i)})

Where :math:`r` is the root of the tree and :math:`\pi(i)` the parent of
attribute :math:`i`.

Learning the Tree
-----------------

The structure is learned with the Chow-Liu procedure [7]_:

1. Compute the conditional mutual information :math:`I(A_i; A_j | C)` between
   every pair of attributes.
2. Build the maximum weight spanning tree of the complete graph with these
   weights.
3. Choose a root (``root``) and direct every edge away from it.

The choice of root does not change which attributes are connected, only the
direction of the edges.

Mixed Data
----------

TAN uses the same strategy as the AnDE family. Continuous attributes are
discretized (``n_bins``, ``strategy``) to compute the mutual information and
when they act as parents; as children they remain continuous and are modeled
by Gaussian distributions. Every conditional :math:`P(x_i | y, x_{\pi(i)})` is
a :class:`skbn.MixedNB` fitted on the augmented class :math:`(y, x_{\pi(i)})`,
so discrete children use Laplace smoothing (``alpha``).

**Key Parameters:**

* ``root``: Index of the root attribute. Default: 0
* ``alpha``: Laplace smoothing for discrete attributes. Default: 1.0
* ``n_bins``: Discretization bins for continuous attributes
* ``strategy``: Discretization strategy ('uniform', 'quantile', 'kmeans')

**Example:**

.. code-block:: python

    from skbn import TAN

    clf = TAN()
    clf.fit(X, y)

    clf.parents_                   # parent of each attribute (-1 for the root)
    clf.conditional_mutual_info_   # edge weights used to learn the tree


.. _kdb:

KDB: K-Dependence Bayesian Classifier
=====================================

:class:`skbn.KDB` [8]_ generalizes Naive Bayes and TAN with a single
parameter: every attribute has the class and **at most k other attributes**
as parents.

.. math::

    P(y, \mathbf{x}) = P(y) \prod_{i} P(x_i | y, \mathbf{x}_{\Pi(i)}), \qquad |\Pi(i)| \leq k

Where :math:`\Pi(i)` is the set of attribute parents of attribute :math:`i`.
With ``k=0`` the model is Naive Bayes.

Learning the Structure
----------------------

The structure is learned greedily [8]_:

1. Compute the mutual information :math:`I(A_i; C)` between every attribute
   and the class, and the conditional mutual information
   :math:`I(A_i; A_j | C)` between every pair of attributes.
2. Add the attributes to the network in decreasing order of
   :math:`I(A_i; C)`.
3. Each attribute takes as parents the :math:`\min(k, t)` attributes, among
   the :math:`t` already added, with the highest :math:`I(A_i; A_j | C)`.

Mixed data is handled as in TAN: continuous attributes are discretized
(``n_bins``, ``strategy``) for the structure and when they act as parents,
and every conditional :math:`P(x_i | y, \mathbf{x}_{\Pi(i)})` is a
:class:`skbn.MixedNB` fitted on the augmented class
:math:`(y, \mathbf{x}_{\Pi(i)})`.

Choosing k
----------

Larger values of ``k`` capture interactions between more attributes, but the
number of cells of the conditional probability tables grows exponentially
with ``k``, so they need more data to be estimated. ``k=1`` (the default)
gives a network of the same complexity as TAN; ``k=2`` or more pays off with
large training sets. Choose ``k`` by cross-validation.

**Key Parameters:**

* ``k``: Maximum number of attribute parents. Default: 1
* ``alpha``: Laplace smoothing for discrete attributes. Default: 1.0
* ``n_bins``: Discretization bins for continuous attributes
* ``strategy``: Discretization strategy ('uniform', 'quantile', 'kmeans')

**Example:**

.. code-block:: python

    from skbn import KDB

    clf = KDB(k=2)
    clf.fit(X, y)

    clf.feature_order_             # order in which the attributes were added
    clf.parents_                   # attribute parents of each attribute
    clf.mutual_info_               # I(A_i; C), used to order the attributes
    clf.conditional_mutual_info_   # I(A_i; A_j | C), used to choose parents


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

``l2_reg`` is the strength of the penalty :math:`\lambda \|\mathbf{w} - 1\|^2`
added to the *summed* conditional log-likelihood, so its effect weakens as the
training set grows. The default of :class:`skbn.WeightedAnDE` is 1.0.
Component weights are few and barely react to it; factor weights are many
and overfit small datasets with a weak penalty, so for them values between
0.1 and 10, chosen by cross-validation, are the safe range.

Computational Considerations
----------------------------

* Use ``n_jobs=-1`` to parallelize SPODE fitting
* The cost of an optimization step grows with the number of weights each
  sample touches: one per component with component weights, one per factor
  with factor weights and one per n-join in ALR
* A2DE with n_features=50 creates ~1,225 sub-models


References
==========

.. [1] Webb, G. I., Boughton, J., & Wang, Z. (2005). Not so naive Bayes:
       Aggregating one-dependence estimators. Machine Learning, 58(1), 5-24.

.. [2] Webb, G. I., Boughton, J., Zheng, F., Ting, K. M., & Salem, H. (2011).
       Learning by extrapolation from marginal to full-multivariate probability
       distributions: Decreasingly naive Bayesian classification. Machine Learning, 86(2), 233-272.

.. [3] Zaidi, N. A., Webb, G. I., Carman, M. J., Petitjean, F., Buntine, W.,
       Hynes, M., & De Sterck, H. (2017). Efficient parameter learning of
       Bayesian network classifiers. Machine Learning, 106(9-10), 1289-1329.

.. [4] Zaidi, N. A., Webb, G. I., Carman, M. J., Petitjean, F., & Cerquides, J. (2016).
       ALR^n: Accelerated higher-order logistic regression. Machine Learning, 104(2-3), 151-194.

.. [5] Zheng, X., Lin, Z., Xu, H., Chen, C., & Ye, T. (2015). Efficient
       learning ensemble SuperParent-one-dependence estimator by maximizing
       conditional log likelihood. Expert Systems with Applications, 42,
       7732-7745.

.. [6] Friedman, N., Geiger, D., & Goldszmidt, M. (1997). Bayesian network
       classifiers. Machine Learning, 29(2-3), 131-163.

.. [7] Chow, C. K., & Liu, C. N. (1968). Approximating discrete probability
       distributions with dependence trees. IEEE Transactions on Information
       Theory, 14(3), 462-467.

.. [8] Sahami, M. (1996). Learning limited dependence Bayesian classifiers.
       In Proceedings of the Second International Conference on Knowledge
       Discovery and Data Mining (KDD-96), 335-338.
