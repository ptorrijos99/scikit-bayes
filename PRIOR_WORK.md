# Prior Work and Related Software

`scikit-bayes` implements semi-naive Bayesian network classifiers behind the scikit-learn estimator interface. This document lists the software that implements the same family of models and what `scikit-bayes` adds to it. The software itself has not been the main content of an earlier peer-reviewed paper; it was the single implementation used in the comparative study of the AnDE family by Torrijos, Gámez and Puerta and described in Chapter 7 of the PhD thesis of P. Torrijos (Universidad de Castilla-La Mancha, 2026).

## Related implementations

| Implementation | Language and API | Models | Continuous attributes | Discriminative parameters |
|---|---|---|---|---|
| Weka AnDE package (Webb et al.) | Java, Weka | A1DE, A2DE | Discretised beforehand | No |
| ALR code (Zaidi et al., 2016) | Java on Weka | AnJE, ALR, wdAnDE (orders 0 to 2) | Discretised beforehand | ALR |
| bnclassify (Mihaljević et al., 2018) | R | NB, TAN, KDB, AODE (order 1), wrapper searches | Discretised beforehand | AWNB, WANBIA, MANB for Naive Bayes |
| BayesNet (Montañana) | C++17 with libtorch, own API | TAN, KDB, SPODE, SPnDE, AODE, A2DE, boosted AODE and A2DE, variants with local discretisation | Discretised, globally or locally (MDLP) | No (boosting reweights instances and models) |
| endgame-ml (Hamilton) | Python, scikit-learn style | TAN, KDB, ensembles of KDB | Discretised internally | No |
| bayesclass (Montañana) | Python on pgmpy, inactive since 2023 | TAN, KDB, SPODE, AODE, BoostAODE | Discretised beforehand or locally | No |
| pgmpy | Python, own API | Naive Bayes, TAN structure search | Discrete | No |

## What `scikit-bayes` adds

1. **The whole AnDE family in one place.** AnDE at any order, with arithmetic or geometric pooling, the n-join estimators (AnJE), and the hybrid generative-discriminative models: ALR and `WeightedAnDE`, with weights per component or per factor at four granularities, fitted jointly or one component at a time. No other Python library provides AnDE beyond order 1 or any of the hybrid models.
2. **Mixed data without a discretisation step.** Every component is a Naive Bayes model on an augmented class, so continuous children keep Gaussian conditionals at any order and only the super-parents are discretised. The supervised MDL and decision-tree discretisers are available as transformers for those who prefer to discretise everything.
3. **The scikit-learn interface.** Every estimator passes `sklearn.utils.estimator_checks.check_estimator` and works inside `Pipeline`, `GridSearchCV` and the rest of the ecosystem. The only dependencies are NumPy, SciPy, Numba and scikit-learn.
4. **TAN and KDB on the same footing.** Both reuse the same estimators and the same handling of mixed data and unseen values, so the structured models and the ensembles can be compared under one implementation.

BayesNet offers boosting and local discretisation, which `scikit-bayes` does not implement, and its C++ kernels are faster on large data.
