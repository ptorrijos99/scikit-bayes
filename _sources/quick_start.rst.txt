.. _quick_start:

###############
Getting Started
###############

Installation
============

From PyPI
---------

scikit-bayes needs Python 3.10 or later, NumPy, SciPy, Numba and
scikit-learn 1.6 or later. Install it from PyPI with pip:

.. prompt:: bash $

    pip install scikit-bayes

From GitHub
-----------

The development version:

.. prompt:: bash $

    pip install git+https://github.com/ptorrijos99/scikit-bayes.git


First models
============

MixedNB on mixed data
---------------------

:class:`skbn.MixedNB` takes Gaussian (continuous), categorical and Bernoulli
(binary) features in one table, and infers the type of each one from its
values:

.. code-block:: python

    import numpy as np
    from skbn import MixedNB

    # Columns: Gaussian, Bernoulli, categorical
    X = np.array([
        [0.5, 0, 0],
        [-1.2, 1, 1],
        [0.6, 1, 2],
        [-0.1, 0, 0],
    ])
    y = np.array([0, 1, 1, 0])

    clf = MixedNB()
    clf.fit(X, y)
    print(clf.predict([[-0.5, 1, 1]]))  # [1]


AnDE on XOR
-----------

:class:`skbn.AnDE` relaxes the independence assumption of Naive Bayes, so it
learns dependencies between features such as XOR, where the class depends
on whether the two features have the same sign:

.. code-block:: python

    import numpy as np
    from skbn import AnDE

    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    y = np.array([0, 1, 1, 0])  # 0 for the same sign, 1 otherwise

    # MixedNB predicts one class everywhere; AnDE gets all four right
    clf = AnDE(n_dependence=1, n_bins=2)
    clf.fit(X, y)
    print(clf.predict(X))  # [0 1 1 0]


Development setup
=================

The development environments are managed with `pixi <https://pixi.sh>`_.
After `installing pixi <https://pixi.sh/latest/#installation>`_, the common
tasks are:

Run tests:

.. prompt:: bash $

    pixi run test

Run linter:

.. prompt:: bash $

    pixi run lint

Build documentation:

.. prompt:: bash $

    pixi run build-doc

Activate development environment:

.. prompt:: bash $

    pixi shell -e dev

The last one opens a shell with the dependencies of the tests, the linter
and the documentation.


Further reading
===============

* :ref:`User guide <user_guide>`: how each model is defined and estimated,
  and how to choose its parameters.
* :ref:`API reference <api>`: parameters, attributes and methods of every
  class.
* :ref:`Example gallery <general_examples>`: what each model captures, shown
  on small datasets with plots.
