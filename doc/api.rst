.. _api:

#############
API Reference
#############

Every class below can be imported from ``skbn``.

.. currentmodule:: skbn

Bayesian Classifiers
====================

Mixed Naive Bayes
-----------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   MixedNB

.. _ande_api:

AnDE Family
-----------

Averaged n-Dependence Estimators and their discriminative weighting.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   AnDE
   WeightedAnDE

AnJE Family
-----------

Averaged n-Join Estimators and Accelerated Logistic Regression, on
discretized attributes or, with Gaussian children, on the factors of the
SPnDEs.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   AnJE
   ALR
   GaussianAnJE
   GaussianALR

.. _tan_api:

Tree Augmented Naive Bayes
--------------------------

Naive Bayes extended with a tree of dependencies between the attributes.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   TAN
   WeightedTAN

.. _kdb_api:

K-Dependence Bayesian Classifier
--------------------------------

Bayesian network in which every attribute has at most k attribute parents.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   KDB
   WeightedKDB


Discretization
==============

Supervised discretizers of continuous attributes, to place before a model in
a pipeline.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   MDLPDiscretizer
   DecisionTreeDiscretizer


Utilities
=========

.. autosummary::
   :toctree: generated/
   :template: function.rst

   show_versions
   check_installation
