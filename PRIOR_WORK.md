# Prior Work and Related Software

`scikit-bayes` implements semi-naive Bayesian network classifiers behind the scikit-learn estimator interface. This document lists the software that implements the same family of models and what `scikit-bayes` adds to it. The software itself has not been the main content of an earlier peer-reviewed paper; it was the single implementation used in the comparative study of the AnDE family by Torrijos, Gámez and Puerta and described in Chapter 7 of the PhD thesis of P. Torrijos (Universidad de Castilla-La Mancha, 2026).

## Related implementations

| Implementation | Language and API | Models | Continuous attributes | Discriminative parameters |
|---|---|---|---|---|
| Weka AnDE package (Webb et al.) | Java, Weka | A1DE, A2DE | Discretised beforehand | No |
| ALR code (Zaidi et al., 2016) | Java on Weka | AnJE, ALR, wdAnDE (orders 0 to 2) | Discretised beforehand | ALR |
| bnclassify (Mihaljević et al., 2018) | R | NB, TAN (Chow-Liu and hill climbing), FSSJ, BSEJ, KDB, AODE (order 1) | Discretised beforehand | AWNB, WANBIA, MANB for Naive Bayes |
| BayesNet (Montañana) | C++17 with libtorch, own API | TAN, KDB, SPODE, SPnDE, AODE, A2DE, boosted AODE and A2DE, variants with local discretisation | Discretised, globally or locally (MDLP) | No (boosting reweights instances and models) |
| endgame-ml (Hamilton) | Python, scikit-learn style | TAN, KDB, ensembles of KDB | Discretised internally | No |
| bayesclass (Montañana) | Python on pgmpy, inactive since 2023 | TAN, KDB, SPODE, AODE, BoostAODE | Discretised beforehand or locally | No |
| pgmpy (Ankan and Textor, 2024) | Python, own API | Naive Bayes, TAN structure search | Discrete | No |
| pyAgrum, `BNClassifier` (Ducamp et al., 2020) | Python on the C++ library aGrUM, scikit-learn style | Naive Bayes, TAN, Chow-Liu tree and general structure search (MIIC, greedy hill climbing, tabu) | Discretised internally | No |
| pomegranate (Schreiber, 2018) | Python on PyTorch, scikit-learn style | Bayes classifier over its probability distributions | Kept continuous | No |
| scikit-learn | Python | Naive Bayes, one attribute type per model (Gaussian, categorical, Bernoulli, multinomial) | Gaussian, in a model of its own | No |
| mixed-naive-bayes | Python, scikit-learn style | Naive Bayes with Gaussian and categorical attributes | Gaussian | No |

## What `scikit-bayes` adds

1. **The whole AnDE family in one place.** AnDE at any order, with arithmetic or geometric pooling, the n-join estimators (AnJE), and the hybrid generative-discriminative models: ALR and `WeightedAnDE`, with weights per component or per factor at four granularities, fitted jointly or one component at a time. To our knowledge no other library with the scikit-learn interface provides AnDE beyond order 1, and none in Python provides the hybrid models; A2DE is otherwise available in Weka (Java) and BayesNet (C++), and ALR in the Java code of its authors.
2. **Mixed data without a discretisation step.** Every component is a Naive Bayes model on an augmented class, so continuous children keep Gaussian conditionals at any order and only the super-parents are discretised. The supervised MDL and decision-tree discretisers are available as transformers for those who prefer to discretise everything.
3. **The scikit-learn interface.** Every estimator passes `sklearn.utils.estimator_checks.check_estimator` and works inside `Pipeline`, `GridSearchCV` and the rest of the ecosystem. The only dependencies are NumPy, SciPy, Numba and scikit-learn.
4. **TAN and KDB on the same footing.** Both reuse the same estimators and the same handling of mixed data and unseen values, so the structured models and the ensembles can be compared under one implementation.

## What it leaves to other software

`scikit-bayes` is a library of semi-naive classifiers, not a general Bayesian network toolkit. Structure search beyond the tree of TAN and the parents of KDB, inference in arbitrary networks and causal analysis are the ground of pyAgrum and pgmpy. BayesNet offers boosting and local discretisation, which `scikit-bayes` does not implement, and its C++ kernels are faster on large data.

## A note on the name

The import name of `scikit-bayes` is `skbn`. pyAgrum has a submodule of the same name, `pyagrum.skbn`, which holds its `BNClassifier` and is often imported as `import pyagrum.skbn as skbn`. The two do not clash, since one is a top-level package and the other a submodule, but the same alias can name either in a script that uses both.
