from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import dump
from skopt import BayesSearchCV
from tpot import TPOTClassifier
import optuna
import numpy as np


class ModelTrainer:
    def __init__(self, model, param_grid=None, search_type='grid', scoring='f1', n_iter=10,
                 save_path='trained_model.joblib'):
        """
        Initialize the model trainer with flexible options for tuning and saving the model.

        Parameters:
        - model: The model to train (any classifier, e.g., RandomForestClassifier, SVC).
        - param_grid (dict): Parameter grid for hyperparameter tuning.
        - search_type (str): Tuning strategy ('grid', 'random', 'bayesian', 'genetic', 'optuna').
        - scoring (str): Scoring metric to optimize during tuning (e.g., 'accuracy', 'f1', 'precision').
        - n_iter (int): Number of parameter settings sampled in RandomizedSearchCV or Bayesian search.
        - save_path (str): File path to save the trained model.
        """
        self.model = model
        self.param_grid = param_grid
        self.search_type = search_type
        self.scoring = scoring
        self.n_iter = n_iter
        self.save_path = save_path
        self.best_model = None

    def objective_optuna(self, trial, x_train, y_train):
        """Objective function for Optuna hyperparameter optimization."""
        # Sample parameters based on param_grid
        params = {key: trial.suggest_categorical(key, values) for key, values in self.param_grid.items()}

        # Set sampled parameters to model and fit
        model = self.model.set_params(**params)
        score = cross_val_score(model, x_train, y_train, scoring=self.scoring, cv=5).mean()

        return score

    def train(self, x_train, y_train):
        """
        Train the model with hyperparameter tuning and save the best model.

        Parameters:
        - X_train: Features of the training set.
        - y_train: Target labels of the training set.

        Returns:
        - best_model: The model with optimal parameters after tuning.
        - best_params: The best parameters identified during tuning.
        """

        # Select tuning strategy
        if self.search_type == 'grid':
            search = GridSearchCV(self.model, self.param_grid, scoring=self.scoring, n_jobs=10, cv=5)

        elif self.search_type == 'random':
            search = RandomizedSearchCV(self.model, self.param_grid, n_iter=self.n_iter,
                                        scoring=self.scoring, n_jobs=10, cv=5)

        elif self.search_type == 'bayesian':
            search = BayesSearchCV(self.model, self.param_grid, n_iter=self.n_iter,
                                   scoring=self.scoring, n_jobs=10, cv=5)

        elif self.search_type == 'genetic':
            tpot = TPOTClassifier(generations=100, population_size=100, verbosity=2,
                                  early_stop=10, n_jobs=10, scoring=self.scoring)
            tpot.fit(x_train, y_train)
            self.best_model = tpot.fitted_pipeline_
            best_params = tpot.fitted_pipeline_.get_params()

        elif self.search_type == 'optuna':
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: self.objective_optuna(trial, x_train, y_train), n_trials=self.n_iter)
            best_params = study.best_params
            self.best_model = self.model.set_params(**best_params)
            self.best_model.fit(x_train, y_train)

        else:
            raise ValueError(
                "search_type must be one of 'grid', 'random', 'bayesian', 'genetic', or 'optuna'.")

        # For search-based tuning strategies
        if self.search_type in ['grid', 'random', 'bayesian']:
            search.fit(x_train, y_train)
            self.best_model = search.best_estimator_
            best_params = search.best_params_

        # Save the best model
        dump(self.best_model, self.save_path)

        return self.best_model, best_params
