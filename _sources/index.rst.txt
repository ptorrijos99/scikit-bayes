.. scikit-bayes documentation master file

:notoc:

################################
scikit-bayes
################################

**Date**: |today| **Version**: |version|

**Useful links**:
`Source Repository <https://github.com/ptorrijos99/scikit-bayes>`__ |
`Issues & Ideas <https://github.com/ptorrijos99/scikit-bayes/issues>`__ |
:ref:`Installation <install>` |
:ref:`Tutorial <sphx_glr_auto_examples_tutorial_00_plot_tutorial.py>` |
:ref:`Citing <citing>` |
`Changelog <https://github.com/ptorrijos99/scikit-bayes/blob/main/CHANGELOG.md>`__ |
`Contributing <https://github.com/ptorrijos99/scikit-bayes/blob/main/CONTRIBUTING.md>`__ |
`Related software <https://github.com/ptorrijos99/scikit-bayes/blob/main/PRIOR_WORK.md>`__

scikit-bayes adds Bayesian network classifiers to scikit-learn. Every model
follows the scikit-learn estimator API and takes tables that mix continuous,
categorical and binary attributes.

* :class:`~skbn.MixedNB`: Naive Bayes for continuous, categorical and binary
  attributes in one model.
* :class:`~skbn.AnDE`: Averaged n-Dependence Estimators (AODE, A2DE).
* :class:`~skbn.AnJE` and :class:`~skbn.GaussianAnJE`: Averaged n-Join
  Estimators.
* :class:`~skbn.TAN` and :class:`~skbn.KDB`: a single network in which every
  attribute has one (TAN) or up to k (KDB) attribute parents.
* :class:`~skbn.WeightedAnDE`, :class:`~skbn.ALR`,
  :class:`~skbn.GaussianALR`, :class:`~skbn.WeightedTAN` and
  :class:`~skbn.WeightedKDB`: weights learned discriminatively on top of the
  generative models.
* :class:`~skbn.MDLPDiscretizer` and :class:`~skbn.DecisionTreeDiscretizer`:
  supervised discretization of continuous attributes.


.. grid:: 1 2 2 2
    :gutter: 4
    :padding: 2 2 0 0
    :class-container: sd-text-center

    .. grid-item-card:: Getting started
        :img-top: _static/img/index_getting_started.svg
        :class-card: intro-card
        :shadow: md

        Installation and a first model.

        .. button-ref:: quick_start
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the getting started guideline

    .. grid-item-card::  User guide
        :img-top: _static/img/index_user_guide.svg
        :class-card: intro-card
        :shadow: md

        How each model is defined and estimated, and how to choose its parameters.

        .. button-ref:: user_guide
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the user guide

    .. grid-item-card::  API reference
        :img-top: _static/img/index_api.svg
        :class-card: intro-card
        :shadow: md

        Parameters, attributes and methods of every class.

        .. button-ref:: api
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the reference guide

    .. grid-item-card::  Examples
        :img-top: _static/img/index_examples.svg
        :class-card: intro-card
        :shadow: md

        What each model captures, shown on small datasets with plots.

        .. button-ref:: general_examples
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the gallery of examples

.. grid:: 1
    :gutter: 4
    :padding: 2 2 0 0
    :class-container: sd-text-center

    .. grid-item-card::  Tutorial
        :class-card: intro-card
        :shadow: md

        Car evaluations, a table of labels, from the DataFrame to the
        fitted network: model comparison, tuning, the learned structure
        against the known one, and evaluation.

        .. button-ref:: sphx_glr_auto_examples_tutorial_00_plot_tutorial.py
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the tutorial


.. toctree::
    :maxdepth: 3
    :hidden:
    :titlesonly:

    install
    quick_start
    Tutorial <auto_examples/tutorial/00_plot_tutorial>
    user_guide
    api
    auto_examples/index
    citing
    Changelog <https://github.com/ptorrijos99/scikit-bayes/blob/main/CHANGELOG.md>
    Contributing <https://github.com/ptorrijos99/scikit-bayes/blob/main/CONTRIBUTING.md>
    Related software <https://github.com/ptorrijos99/scikit-bayes/blob/main/PRIOR_WORK.md>
