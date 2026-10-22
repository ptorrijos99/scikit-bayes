.. _api:

#############
API Reference
#############

This is the API documentation for the scikit-bayes package.

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

Averaged n-Join Estimators and Accelerated Logistic Regression.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   AnJE
   ALR

.. _tan_api:

Tree Augmented Naive Bayes
--------------------------

Naive Bayes extended with a tree of dependencies between the attributes.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   TAN


Utilities
=========

.. autosummary::
   :toctree: generated/
   :template: functions.rst

     utils.discovery.all_estimators
     utils.discovery.all_displays
     utils.discovery.all_functions
