# Prior Work and Ecosystem Gap

`scikit-bayes` gives Python developers a way to use Bayesian network classifiers and probabilistic ensembles without leaving the scikit-learn ecosystem.

## Existing Tools

If you want to use Bayesian methods in Python today, you generally have three options:

1. **scikit-learn (`sklearn.naive_bayes`)**: This includes standard Naive Bayes (Gaussian, Multinomial, Categorical, Complement, Bernoulli). It is fast and easy to use, but strictly assumes conditional independence between features. If your dataset has correlated features, standard Naive Bayes will struggle. You also have to handle continuous and categorical data separately using `ColumnTransformer`s.
2. **pgmpy**: A large library for Probabilistic Graphical Models. You can use it to build custom Bayesian Networks and run exact or approximate inference. However, it is not built for fast, tabular classification tasks, and its API does not match scikit-learn.
3. **bnlearn**: A Python wrapper around R's `bnlearn` and `pgmpy`. People mostly use it to discover DAG structures and plot them, rather than to train predictive classifiers for production.

## How `scikit-bayes` fits in

`scikit-bayes` combines the speed and API of scikit-learn with the capabilities of Bayesian Networks. It focuses specifically on **Averaged n-Dependence Estimators (AnDE)** and their continuous extensions.

1. **Relaxing the Independence Assumption**: Standard Naive Bayes assumes features don't interact. By implementing AnDE and AnJE, `scikit-bayes` lets you condition on $n$ features simultaneously. This captures correlations without forcing you to learn a full Bayesian Network structure.
2. **Mixed Data Handling**: The `MixedNB` and AnDE models handle datasets with both continuous and categorical features out of the box. They use supervised decision tree discretizers and kernel density estimators internally, so you don't have to build complex preprocessing pipelines.
3. **Hybrid Generative-Discriminative Models**: Generative models are interpretable, but discriminative models usually predict better. `scikit-bayes` includes `ALR` (Accelerated Logistic Regression) and `WeightedAnDE`, which optimize generative structures using log-linear discriminative methods.
4. **Scikit-learn Compatibility**: Every estimator passes `sklearn.utils.estimator_checks.check_estimator`. You can plug them directly into any `Pipeline` or `GridSearchCV`.
