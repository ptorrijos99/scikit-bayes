.. title:: User guide : contents

.. _user_guide:

==========
User Guide
==========

This guide describes how each model of scikit-bayes is defined and
estimated, and how to choose its parameters.

.. contents:: Table of Contents
   :local:
   :depth: 2

Introduction
============

Bayesian network classifiers
----------------------------

Bayesian network classifiers (BNCs) compute class probabilities with Bayes' theorem. The simplest is Naive Bayes, which assumes the features independent given the class:

.. math::

    P(y|\mathbf{x}) \propto P(y) \prod_{i=1}^{n} P(x_i|y)

The assumption rarely holds, but Naive Bayes scales well and is a sound baseline. When the features depend strongly on each other, as in a parity problem like XOR, its probabilities are wrong.

What the library adds to scikit-learn
-------------------------------------

The Naive Bayes classes of scikit-learn each take one type of feature, so a table with continuous, categorical and binary columns has to be split with a ``ColumnTransformer`` and the probabilities of the parts combined by hand; :class:`skbn.MixedNB` takes the whole table. Beyond Naive Bayes, scikit-learn has no Bayesian network classifier that models dependencies between features. scikit-bayes adds the AnDE family (AODE, A2DE), TAN and KDB, and the models that learn discriminative weights on top of them, such as ALR and WeightedAnDE. Every estimator passes the scikit-learn estimator checks, so all of them work in pipelines, cross-validation and grid search.


.. _mixed_naive_bayes:

MixedNB: Naive Bayes for mixed data
===================================

:class:`skbn.MixedNB` handles datasets with heterogeneous feature types
by internally combining scikit-learn's specialized Naive Bayes estimators.

A table of patients may hold the age (continuous, Gaussian), the sex
(binary, Bernoulli) and the education level coded 0 to 3 (categorical). With
scikit-learn this takes three Naive Bayes classifiers, one per type of
feature, and a sum of their log-probabilities. MixedNB fits one model on the
whole table.

Feature types
-------------

MixedNB types every feature not declared by the user during ``fit()``, from
its training values. Every other estimator of the library uses the same rule
through its class-level :class:`~skbn.MixedNB`, so a feature has the same type
as a child and as a parent:

* **Ignored**: constant features, which carry no information.
* **Bernoulli**: features with exactly 2 unique values.
* **Categorical**: non-negative integer-valued features with 3 to 20 unique
  values.
* **Gaussian**: everything else, that is, fractional or negative values and
  integer values with more than 20 distinct values (ages, counts, amounts).

The rule only looks at the values, so declaring the types is the safe choice
whenever they are known, either as indices or as a boolean mask:

.. code-block:: python

    from skbn import MixedNB

    # Force features 2 and 3 to be categorical
    clf = MixedNB(categorical_features=[2, 3])

    # Force feature 1 to be Bernoulli
    clf = MixedNB(bernoulli_features=[1])

    # The same as the first one, with a boolean mask
    clf = MixedNB(categorical_features=[False, False, True, True])

Each type is estimated as follows:

* **Categorical** features are label-encoded on the values seen in training,
  so the codes may be any numbers (negative, sparse or non-integer). A value
  never seen in training contributes nothing to the prediction, as a missing
  value would.
* **Bernoulli** features are binarised as ``x > t``. When the training values
  lie on both sides of 0, ``t = 0`` as in
  :class:`~sklearn.naive_bayes.BernoulliNB`; otherwise ``t`` is the midpoint of
  their range, so that two values such as {1, 2} stay apart.
* **Gaussian** features with zero variance in training are ignored.

Labels and pandas categories
----------------------------

Every estimator of the library also takes columns of labels: pandas
``category`` columns, whatever their categories, and columns of strings (in a
DataFrame, or in a NumPy object or string array). Such a column is
categorical whatever its values, even a ``category`` column with more than 20
integer categories; it may be declared Bernoulli but not Gaussian, and it is
ignored when constant. Its labels are coded by their position among the
sorted labels seen in training, so the model is exactly the one fitted on
these codes declared categorical. The order of the categories of a
``category`` dtype is not used, and its categories absent from the training
rows get no estimate. A label never seen in training is an unseen category,
and :attr:`~skbn.MixedNB.categories_` holds the labels of each column.

.. code-block:: python

    import pandas as pd
    from skbn import MixedNB

    X = pd.DataFrame({
        "age": [23.0, 45.5, 31.2, 52.8, 38.1, 27.4],
        "plan": pd.Categorical(["basic", "pro", "pro", "basic", "pro", "basic"]),
        "city": ["Madrid", "Paris", "Paris", "Rome", "Madrid", "Rome"],
    })
    clf = MixedNB().fit(X, [0, 1, 1, 0, 1, 0])
    print(clf.feature_types_)
    # {'gaussian': [0], 'categorical': [1, 2], 'bernoulli': []}

A column that holds labels in ``fit`` must hold labels in ``predict`` (strings
and categories may be swapped), and the other columns must hold numbers. The
estimators do not take missing values yet: impute them first, for example
with :class:`~sklearn.impute.SimpleImputer`.

Example
-------

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
    # {'gaussian': [0], 'categorical': [2], 'bernoulli': [1]}

    # Predict
    print(clf.predict([[1.0, 1, 1]]))  # [1]
    print(clf.predict_proba([[1.0, 1, 1]]))  # [[0.14 0.86]]

Parameters
----------

* ``alpha``: Laplace smoothing of the categorical and Bernoulli features. Default: 1.0
* ``var_smoothing``: variance smoothing of the Gaussian features. Default: 1e-9
* ``categorical_features``: indices or boolean mask of the categorical features
* ``bernoulli_features``: indices or boolean mask of the Bernoulli features
* ``gaussian_features``: indices or boolean mask of the Gaussian features


The AnDE family
===============

The Averaged n-Dependence Estimators (AnDE) relax the independence
assumption by conditioning every feature on the class and on ``n``
"super-parent" features.

Where Naive Bayes fails
-----------------------

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

Naive Bayes cannot learn this problem, and its accuracy stays at chance level.
AnDE models P(X2 | Y, X1) instead of P(X2 | Y), and the dependence on X1 is
what separates the classes.

Super-parents
-------------

An SPnDE (Super-Parent n-Dependence Estimator) conditions every child
feature on the class and on ``n`` parent features:

.. math::

    P(y, \mathbf{x}) = P(Y^*) \prod_{i \in children} P(x_i | Y^*)

where :math:`Y^* = (y, x_{p1}, x_{p2}, ..., x_{pn})` is the augmented
super-class. AnDE averages the SPnDEs of every set of ``n`` parents.

AnDE
----

:class:`skbn.AnDE` is the standard generative model described by Webb et al. [1]_ [2]_.

.. math::

    P(y|\mathbf{x}) \propto \sum_{m} P_m(y, \mathbf{x})

Main parameters:

* ``n_dependence``: order of dependence. ``0`` is Naive Bayes
  (:class:`skbn.MixedNB`), ``1`` is AODE and ``2`` is A2DE.
* ``n_bins``: number of bins of the continuous super-parents.
* ``strategy``: how those bins are placed (``'uniform'``, ``'quantile'``,
  ``'kmeans'``, or ``'mdlp'``, which also chooses how many: see
  :ref:`discretization`).

Example:

.. code-block:: python

    from skbn import AnDE

    # AODE (n=1)
    clf = AnDE(n_dependence=1, n_bins=5)
    clf.fit(X, y)

    # A2DE (n=2): more components, and more data needed per estimate
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
``n_bins`` bins. The paper uses MDL discretization instead, which is
``strategy="mdlp"`` (see :ref:`discretization`). Probabilities are m-estimates
with ``m = alpha``. A value never seen in training says nothing about the
class, so the n-joins that contain it are left out for that sample.

``weight_level=3`` (the default) is the model of the paper. Levels 1 (one
weight per n-join) and 2 (per n-join and value) tie the weights following the
nesting of [3]_. The other defaults reproduce the settings of the code
released with the paper: weights start at ``w = 1``, there is no penalty, and
L-BFGS-B keeps 5 corrections and runs until the relative reduction of the
objective falls below 1e7 machine epsilons (its ``factr = 1e7``) or for
10\ :sup:`4` iterations, with the test on the projected gradient disabled.
The starting point is not the AnJE but the unaveraged
product of all n-joins, so on small samples the unpenalized model overfits.
``l2_reg`` adds the penalty towards ``w = 1`` that the paper suggests for
small datasets, and ``max_iter`` and ``tol`` stop the optimizer earlier.

.. code-block:: python

    from skbn import ALR, AnJE

    clf = AnJE(n_join=2)
    clf = ALR(n_join=2)  # as in Zaidi et al. (2016)
    clf.fit(X, y)

Gaussian children
-----------------

By the chain rule every (n+1)-join is a product of the factors of an SPnDE,
:math:`P(x_\alpha | y) = P(x_P | y)\, P(x_j | y, x_P)` for each
:math:`j \in \alpha` and :math:`P = \alpha \setminus \{j\}`. Averaging these
orders and weighting the joins as in AnJE gives a log-linear pool of the
factors of the SPnDEs of order ``n``,

.. math::

    \log P(y|\mathbf{x}) \propto \log P(y) + \frac{1}{|M|} \sum_{P}
    \Big[ \tfrac{d}{n+1} \log P(\mathbf{x}_P | y)
    + \tfrac{d}{(n+1)(d-n)} \sum_{j \notin P} \log P(x_j | y, \mathbf{x}_P) \Big].

:class:`skbn.GaussianAnJE` builds AnJE on these factors, and
:class:`skbn.GaussianALR` puts one weight per factor, value, class and parent
instantiation on them, as ALR\ :sup:`n` does on the n-joins. Since the factors
are those of :class:`skbn.AnDE`, both models handle mixed data as HAODE does:
only the super-parents are discretized, and continuous children keep their
Gaussian conditionals. ``n_dependence`` is the order of the SPnDEs, so
``GaussianAnJE(n_dependence=1)`` is the counterpart of ``AnJE(n_join=2)``.
Probabilities are Laplace estimates on the augmented states, and the
super-parent factor is divided by its total within the class so that every
factor is conditional on the class. On discrete data the two versions
coincide as the smoothing vanishes, and ``GaussianALR`` at ``w = 1`` is
``GaussianAnJE``.

.. code-block:: python

    from skbn import GaussianALR, GaussianAnJE

    clf = GaussianAnJE(n_dependence=1)  # A2JE with Gaussian children
    clf = GaussianALR(n_dependence=1)  # ALR^2 with Gaussian children
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

Learning the tree
-----------------

The structure is learned with the Chow-Liu procedure [7]_:

1. Compute the conditional mutual information :math:`I(A_i; A_j | C)` between
   every pair of attributes.
2. Build the maximum weight spanning tree of the complete graph with these
   weights.
3. Choose a root (``root``) and direct every edge away from it.

The choice of root does not change which attributes are connected, only the
direction of the edges.

Mixed data
----------

TAN handles mixed data as the AnDE family does. Continuous attributes are
discretized (``n_bins``, ``strategy``) to compute the mutual information and
when they act as parents; as children they remain continuous and are modeled
by Gaussian distributions. Every conditional :math:`P(x_i | y, x_{\pi(i)})` is
Naive Bayes on the augmented class :math:`(y, x_{\pi(i)})`, with the
estimates of :class:`skbn.MixedNB`, so discrete children use Laplace
smoothing (``alpha``).

Main parameters:

* ``root``: index of the root attribute. Default: 0
* ``alpha``: Laplace smoothing of the discrete attributes. Default: 1.0
* ``n_bins``: number of bins of the continuous attributes
* ``strategy``: how those bins are placed (``'uniform'``, ``'quantile'``, ``'kmeans'``, ``'mdlp'``)

Example:

.. code-block:: python

    from skbn import TAN

    clf = TAN()
    clf.fit(X, y)

    clf.parents_                   # parent of each attribute (-1 for the root)
    clf.conditional_mutual_info_   # edge weights used to learn the tree
    clf.edges_                     # the arcs as (parent, child) pairs
    print(clf.to_dot())            # the network, for Graphviz or networkx


.. _kdb:

KDB: K-dependence Bayesian classifier
=====================================

:class:`skbn.KDB` [8]_ generalizes Naive Bayes and TAN with a single
parameter: every attribute has the class and **at most k other attributes**
as parents.

.. math::

    P(y, \mathbf{x}) = P(y) \prod_{i} P(x_i | y, \mathbf{x}_{\Pi(i)}), \qquad |\Pi(i)| \leq k

Where :math:`\Pi(i)` is the set of attribute parents of attribute :math:`i`.
With ``k=0`` the model is Naive Bayes.

Learning the structure
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
and every conditional :math:`P(x_i | y, \mathbf{x}_{\Pi(i)})` is Naive Bayes
on the augmented class :math:`(y, \mathbf{x}_{\Pi(i)})`.

Choosing k
----------

Larger values of ``k`` capture interactions between more attributes, but the
number of cells of the conditional probability tables grows exponentially
with ``k``, so they need more data to be estimated. ``k=1`` (the default)
gives a network of the same complexity as TAN; ``k=2`` or more pays off with
large training sets. Choose ``k`` by cross-validation.

Main parameters:

* ``k``: maximum number of attribute parents. Default: 1
* ``alpha``: Laplace smoothing of the discrete attributes. Default: 1.0
* ``n_bins``: number of bins of the continuous attributes
* ``strategy``: how those bins are placed (``'uniform'``, ``'quantile'``, ``'kmeans'``, ``'mdlp'``)

Example:

.. code-block:: python

    from skbn import KDB

    clf = KDB(k=2)
    clf.fit(X, y)

    clf.feature_order_             # order in which the attributes were added
    clf.parents_                   # attribute parents of each attribute
    clf.mutual_info_               # I(A_i; C), used to order the attributes
    clf.conditional_mutual_info_   # I(A_i; A_j | C), used to choose parents
    clf.edges_                     # the arcs as (parent, child) pairs
    print(clf.to_dot())            # the network, for Graphviz or networkx


.. _weighted_networks:

Weighted TAN and KDB
====================

:class:`skbn.WeightedTAN` and :class:`skbn.WeightedKDB` are the weighted
conditionally Bayesian network (wCCBN) of Zaidi et al. [3]_ on the networks
of TAN and KDB. The structure and the probabilities are learned as in
:class:`skbn.TAN` and :class:`skbn.KDB` and stay fixed; every factor of the
joint, the class prior and the conditional of every attribute, gets an
exponent weight:

.. math::

    \log P^w(y | \mathbf{x}) = w_y \log P(y)
    + \sum_{i} w_{i, x_i, y, \mathbf{x}_{\Pi(i)}}
    \log P(x_i | y, \mathbf{x}_{\Pi(i)}) - \log Z(\mathbf{x})

The weights maximise the conditional log-likelihood of the training set with
L-BFGS-B, a convex problem. They start at, and are regularized towards
(``l2_reg``), the generative model ``w = 1``, which gives the predictions of
TAN or KDB. ``weight_level`` ties the weights with the nesting of the factor
weights of :class:`skbn.WeightedAnDE` (see its table above): one weight per
factor (1), per factor and value (2), per factor, value and class (3), or per
factor, value, class and parent instantiation (4). Level 4, one weight per
parameter of the network, is the wCCBN of the paper and the default. The value
of a Gaussian child is not discretized, so its weights ignore the value. Only
the combinations seen in training get a weight; any other one, such as a
parent value never seen, keeps the generative weight 1.

The weights help when the data break the independence assumptions of the
network. A tree or ``k`` parents explain only part of the dependencies
between correlated attributes: the conditional of the copy of an attribute
given that attribute is close to 1 for a discrete copy, but a continuous copy
is Gaussian given the *bin* of its parent and still repeats part of the
evidence. The generative model then counts that evidence several times and is
overconfident; the weights scale the repeated conditionals down. With few
samples per weight, the finer levels overfit unless ``l2_reg`` is large
enough; level 1 or 2 is the safer choice on small datasets.

Every evaluation of the objective visits each factor of each training sample
for every class, :math:`O(n \, (d + 1) \, |C|)` operations, and the fit keeps
these log-likelihoods in memory, about :math:`12\, n\, (d + 1)\, |C|`
bytes (54 MB for 60,000 samples, 24 attributes and 3 classes). Predictions
cost about as much as those of TAN and KDB.

A weighted network is a single component, so it is the geometric
:class:`skbn.WeightedAnDE` with ``weight_scope="factor"`` reduced to one
SPnDE: ``WeightedKDB(k=0)`` and ``WeightedAnDE(n_dependence=0,
weight_scope="factor", mean="geometric")`` fit the same weights (with
``var_prior_weight=0``; otherwise the Gaussian variances of the two Naive
Bayes models differ). WeightedAnDE weights the factors of :math:`d` SPnDEs,
about :math:`d` times more weights and work for the same number of
attributes. :class:`skbn.ALR` is the same idea on the n-joins of AnJE instead
of the conditionals of a network.

.. code-block:: python

    from skbn import WeightedKDB, WeightedTAN

    clf = WeightedTAN()  # wCCBN on the TAN structure
    clf = WeightedKDB(k=2, weight_level=1, l2_reg=10.0)  # one weight per factor
    clf.fit(X, y)

    clf.learned_weights_   # the weights, 1 for the generative model
    clf.prior_weights_     # those of the class prior
    clf.feature_weights_   # those of every attribute: below 1, it is discounted


.. _discretization:

Supervised discretization
=========================

By default the models discretize their continuous attributes with
:class:`~sklearn.preprocessing.KBinsDiscretizer` (``n_bins``, ``strategy``),
which does not look at the class. The entropy discretization of Fayyad and
Irani (1993), the usual choice in the AnDE and ALR literature, places the
cut-points where the class changes instead. It splits each attribute
recursively at the cut-point of least class entropy and stops when the gain
no longer pays for the cut under the MDL criterion, so the number of bins is
not fixed and ``n_bins`` is not used. An attribute without any accepted cut
becomes constant, and the models then ignore it wherever it was to be
discretized.

Every model takes it as ``strategy="mdlp"``. It applies where the model
discretizes: to every continuous attribute in :class:`skbn.AnJE` and
:class:`skbn.ALR`, and to the super-parents and the structure in the other
models, whose continuous children stay Gaussian.

.. code-block:: python

    from skbn import ALR, TAN

    # ALR^2 with the discretization of Zaidi et al. (2016)
    clf = ALR(n_join=2, strategy="mdlp")

    # TAN whose tree is learned on MDL intervals, as in Friedman et al. (1997)
    clf = TAN(strategy="mdlp")
    clf.fit(X, y)

The same discretization is available as a transformer,
:class:`skbn.MDLPDiscretizer`, together with
:class:`skbn.DecisionTreeDiscretizer`, which uses the thresholds of a
decision tree fitted on each attribute, with at most ``n_bins`` leaves. In a
pipeline they discretize every attribute they are given, so the model that
follows has no continuous children.

.. warning::

    Declare the discretized attributes as ``categorical_features`` in the
    model that follows a discretizer. The models type an undeclared attribute
    from its values, and integer codes with more than 20 distinct values are
    taken as a continuous attribute and discretized again. MDL discretization
    produces that many intervals on large samples.

.. code-block:: python

    from sklearn.pipeline import make_pipeline
    from skbn import AnDE, MDLPDiscretizer

    # A1DE on discretized attributes only (here X has 10 columns)
    clf = make_pipeline(
        MDLPDiscretizer(),
        AnDE(n_dependence=1, categorical_features=range(10)),
    )
    clf.fit(X, y)


.. _choosing_a_model:

Choosing a model
================

The table sums up the sections above, with ``d`` attributes and ``n``
training samples. The generative models are fitted by counting; the weighted
ones then run L-BFGS-B on the conditional log-likelihood, starting at the
generative model.

.. list-table::
   :header-rows: 1
   :widths: 16 32 24 28

   * - Model
     - What it assumes or captures
     - Training cost
     - When to use it
   * - :class:`~skbn.MixedNB`
     - Features independent given the class; continuous, categorical and
       binary features in one table.
     - Counting, one estimate per feature and class.
     - A baseline that scales well; fails when features depend strongly on
       each other, as in XOR.
   * - :class:`~skbn.AnDE`, ``n_dependence=1`` (AODE)
     - Every child conditioned on the class and one super-parent, averaged
       over the ``d`` super-parents.
     - Counting, ``d`` components.
     - The usual default: it captures pairwise dependencies and its
       estimates need little data.
   * - :class:`~skbn.AnDE`, ``n_dependence=2`` (A2DE)
     - Every child conditioned on the class and two super-parents.
     - Counting, :math:`d(d-1)/2` components, each estimated on more cells.
     - Large training sets; orders of 3 or more rarely pay for their cost.
   * - :class:`~skbn.WeightedAnDE`
     - The SPnDEs of AnDE with discriminative weights on each component or
       on each factor.
     - AnDE, then L-BFGS-B; a step costs one weight per component, or per
       factor, for each sample.
     - To learn how much each SPnDE is trusted (component weights), or to
       correct a dependence between children that every SPnDE shares, such
       as a duplicated attribute (factor weights). Factor weights overfit
       small datasets with a weak ``l2_reg``.
   * - :class:`~skbn.AnJE` / :class:`~skbn.ALR`
     - Joint distributions of every subset of ``n_join`` attributes given
       the class; ALR adds one weight per n-join, value and class (a convex
       problem). Continuous attributes are discretized.
     - Counting of the :math:`\binom{d}{n}` n-joins; ALR then runs L-BFGS-B
       with one weight per n-join touched by each sample.
     - Discrete data, or data discretized with :class:`~skbn.MDLPDiscretizer`
       as in Zaidi et al. [4]_. Unpenalized ALR overfits small samples;
       ``l2_reg`` adds the penalty the paper suggests for them.
   * - :class:`~skbn.GaussianAnJE` / :class:`~skbn.GaussianALR`
     - The same pool as AnJE, written on the factors of the SPnDEs of order
       ``n_dependence``: only the super-parents are discretized and
       continuous children stay Gaussian.
     - Counting as AnDE of the same order; GaussianALR then runs L-BFGS-B
       with one weight per factor, value, class and parent instantiation.
     - Mixed data where continuous attributes should not lose their values
       as children, as in HAODE.
   * - :class:`~skbn.TAN`
     - A single network in which every attribute has the class and at most
       one other attribute as parents (a tree).
     - Conditional mutual information of every pair of attributes, time
       growing with :math:`n d^2`, then counting.
     - A single network whose structure can be read (``parents_``), instead
       of an average of many; KDB with ``k=1`` has the same complexity.
   * - :class:`~skbn.KDB`
     - A single network in which every attribute has the class and at most
       ``k`` other attributes as parents.
     - As TAN, plus the mutual information with the class; the tables grow
       exponentially with ``k``.
     - Interactions between more than two attributes; ``k=2`` or more pays
       off with large training sets. Choose ``k`` by cross-validation.
   * - :class:`~skbn.WeightedTAN` / :class:`~skbn.WeightedKDB`
     - The network of TAN or KDB with an exponent weight on the class prior
       and on every conditional (see :ref:`weighted_networks`).
     - TAN or KDB, then L-BFGS-B; a step costs one weight per factor for each
       sample and class.
     - Correlated or duplicated attributes that the network counts several
       times. On small samples the weights can overfit: compare with the
       generative model by cross-validation.


Choosing the parameters
=======================

Order of dependence
-------------------

AnDE of order ``n`` has one component for every set of ``n`` attributes,
:math:`\binom{d}{n}` of them over ``d`` attributes, and each component
estimates its children on more cells as ``n`` grows.

* ``n=1`` (AODE) is the usual default: ``d`` components, and its estimates
  need little data.
* ``n=2`` (A2DE) has :math:`d(d-1)/2` components (1,225 with 50
  attributes) and pays off on large training sets. Components whose parents
  are constant in training share their tables, so the cost grows with the
  informative attributes. On a laptop, A2DE fits 90% of Census-Income (41
  attributes) in about 9 s and of Covertype (54 attributes, 520,000 rows) in
  about 40 s.
* ``n>=3`` rarely pays for its cost, which grows combinatorially.

Discretization
--------------

The super-parents of AnDE and the attributes of AnJE, ALR, TAN and KDB are
discretized when continuous:

* ``'quantile'`` (default): bins of equal frequency, robust to outliers.
* ``'uniform'``: bins of equal width.
* ``'kmeans'``: bins around the clusters of one-dimensional k-means.
* ``'mdlp'``: cuts where the class changes, as many as the MDL criterion
  accepts (see :ref:`discretization`).

``n_bins`` is usually between 3 and 10: more bins give finer cuts but fewer
samples per estimate. ``'mdlp'`` chooses the number of bins of each attribute
itself: many on large samples of an informative attribute, a single one for
an attribute that says nothing about the class on its own.

Regularization of the weights
-----------------------------

``l2_reg`` is the strength of the penalty :math:`\lambda \|\mathbf{w} - 1\|^2`
added to the *summed* conditional log-likelihood, so its effect weakens as the
training set grows. The default of :class:`skbn.WeightedAnDE`,
:class:`skbn.WeightedTAN` and :class:`skbn.WeightedKDB` is 1.0.
Component weights are few and barely react to it; factor weights are many
and overfit small datasets with a weak penalty, so for them values between
0.1 and 10, chosen by cross-validation, are the safe range.

.. _computational_considerations:

Computational considerations
----------------------------

* The counting and the evaluation of the SPnDEs, of the n-joins, of the
  weighted objectives and of the conditionals of TAN and KDB run in compiled
  Numba kernels. They are compiled on the first use of each model and cached
  on disk: after installing or upgrading the package, the first fit of a
  model takes longer (on a laptop, a few seconds for MixedNB, AnJE or TAN,
  and 15 to 30 s for AnDE and WeightedAnDE), and later sessions load the
  compiled code in a fraction of a second.
* ``n_jobs`` sets the number of threads of those kernels (1 by default,
  ``-1`` for all processors). The threaded version of a kernel is compiled
  the first time it is used, so a model run with several threads for the
  first time pays that compilation once more. The weights of the weighted
  models are optimized by L-BFGS-B in a single thread, with the objective of
  ALR evaluated in ``n_jobs`` threads.
* TAN and KDB compute the conditional mutual information of every pair of
  attributes: the time grows with :math:`n d^2` and the matrix
  ``conditional_mutual_info_`` holds :math:`d^2` values (800 MB for 10,000
  attributes).
* The parameters are stored only for the augmented states and the cells seen
  in training, so the memory of a model is bounded by the data rather than by
  the cardinalities of the attributes.
* The cost of an optimization step grows with the number of weights each
  sample touches: one per component with component weights, one per factor
  with factor weights and one per n-join in ALR.


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
