"""
============================
Gradient Boosting regression
============================

Demonstrate Gradient Boosting on the Boston housing dataset.

Using sacred to store the experiment in a mongodb
"""

import numpy as np
import matplotlib.pyplot as plt
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
import labwatch
from labwatch.assistant import LabAssistant
from labwatch.hyperparameters import UniformInt, UniformFloat
from labwatch.optimizers.random_search import RandomSearch
import labwatch.optimizers.bayesian_optimization
from labwatch.optimizers.bayesian_optimization import BayesianOptimization
from labwatch.optimizers.random_forest import RandomForestOpt
from labwatch.optimizers.smac_wrapper import SMAC

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

ex = Experiment('gradient_boosting')

#dbname = "labwatch_bo_opt"
dbname = "labwatch_smac"
#dbname = "labwatch_bohamiann"
#dbname = "labwatch_bo_1000"

ex.observers.append(MongoObserver(
    url="127.0.0.1:27017",
    db_name=dbname
))
ex.observers.append(FileStorageObserver(dbname))

a = LabAssistant(ex, optimizer=SMAC)


@ex.config
def cfg(_log):
    n_estimators = 500
    max_depth = 4
    min_samples_split = 2
    learning_rate = 0.01
    loss = 'ls'
    tags = ['gb_fixed_params',]
    _log.debug('Dummy log to fix fallback error.')

@a.search_space
def search_space():
    n_estimators = UniformInt(lower=2, upper=1000, default=250, log_scale=False)
    max_depth = UniformInt(lower=1, upper=20, default=3)
    min_samples_split = UniformInt(lower=2, upper=10, default=2)
    learning_rate = UniformFloat(lower=0.001, upper=0.99, default=.005)
    tags = ['gb', 'labwatch']

@a.search_space
def reduced_search_space():
    n_estimators = UniformInt(lower=300, upper=1000, default=450, log_scale=False)
    max_depth = UniformInt(lower=2, upper=4, default=3)
    min_samples_split = UniformInt(lower=2, upper=5, default=3)
    learning_rate = UniformFloat(lower=0.001, upper=0.15, default=.005)
    tags = ['gb', 'labwatch', 'limited_search_space']

@ex.automain
def run(_run, n_estimators, max_depth, min_samples_split, learning_rate, loss, _seed):
    # Load data
    boston = datasets.load_boston()
    # Fit regression model
    clf = ensemble.GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        learning_rate=learning_rate,
        loss=loss,
    )
    mse = float(abs(np.mean(cross_val_score(
        estimator=clf,
        X=boston.data.astype(np.float32),
        y=boston.target,
        scoring='neg_mean_squared_error',
        cv=KFold(n_splits=10, shuffle=True, random_state=_seed)
        ))))

    _run.log_scalar('clf.MSE', mse)

    # Plot training deviance

    # compute test set deviance
#    test_score = np.zeros((n_estimators,), dtype=np.float64)
#
#    for i, y_pred in enumerate(clf.staged_predict(X_test)):
#        test_score[i] = clf.loss_(y_test, y_pred)
#
#    fig = plt.figure(figsize=(12, 6))
#    plt.subplot(1, 2, 1)
#    plt.title('Deviance')
#    plt.plot(np.arange(n_estimators) + 1, clf.train_score_, 'b-',
#            label='Training Set Deviance')
#    plt.plot(np.arange(n_estimators) + 1, test_score, 'r-',
#            label='Test Set Deviance')
#    plt.legend(loc='upper right')
#    plt.xlabel('Boosting Iterations')
#    plt.ylabel('Deviance')
#
#    # #############################################################################
#    # Plot feature importance
#    feature_importance = clf.feature_importances_
#    # make importances relative to max importance
#    feature_importance = 100.0 * (feature_importance / feature_importance.max())
#    sorted_idx = np.argsort(feature_importance)
#    pos = np.arange(sorted_idx.shape[0]) + .5
#    plt.subplot(1, 2, 2)
#    plt.barh(pos, feature_importance[sorted_idx], align='center')
#    plt.yticks(pos, boston.feature_names[sorted_idx])
#    plt.xlabel('Relative Importance')
#    plt.title('Variable Importance')
#    # plt.show()

    #_run.add_artifact(fig)
    results = dict()
    results['optimization_target'] = mse

    return results
