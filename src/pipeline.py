# pipeline.py

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from feature_selection import FeatureSelector
from model_training import ModelTrainer
from evaluation import ModelEvaluator
from joblib import load


class MLPipeline:
    def __init__(self, config):
        """
        Initialize the pipeline with configuration for each stage.

        Parameters:
        - config (dict): Configuration dictionary with parameters for each stage of the pipeline.
        """
        # Initialize each module based on provided configuration
        self.data_preprocessor = DataPreprocessor(
            continuous_features=config['preprocessing']['continuous_features'],
            categorical_features=config['preprocessing']['categorical_features'],
            binary_features=config['preprocessing']['binary_features'],
            ordinal_features=config['preprocessing']['ordinal_features'],
            ordinal_order=config['preprocessing']['ordinal_order'],
            outlier_threshold=config['preprocessing'].get('outlier_threshold', 3),
            skew_threshold=config['preprocessing'].get('skew_threshold', 0.5),
            balance_method=config['preprocessing'].get('balance_method', None)
        )

        self.feature_engineer = FeatureEngineer(
            interaction_features=config['feature_engineering'].get('interaction_features', []),
            polynomial_degree=config['feature_engineering'].get('polynomial_degree', None),
            binning_features=config['feature_engineering'].get('binning_features', []),
            n_bins=config['feature_engineering'].get('n_bins', 5),
            binning_strategy=config['feature_engineering'].get('binning_strategy', 'uniform'),
            pca_components=config['feature_engineering'].get('pca_components', None)
        )

        self.feature_selector = FeatureSelector(
            variance_threshold=config['feature_selection'].get('variance_threshold', None),
            correlation_threshold=config['feature_selection'].get('correlation_threshold', None),
            use_rfe=config['feature_selection'].get('use_rfe', False),
            rfe_model=config['feature_selection'].get('rfe_model', None),
            use_lasso=config['feature_selection'].get('use_lasso', False),
            lasso_alpha=config['feature_selection'].get('lasso_alpha', 1.0),
            use_tree_importance=config['feature_selection'].get('use_tree_importance', False),
            tree_model=config['feature_selection'].get('tree_model', None),
            n_features=config['feature_selection'].get('n_features', None)
        )

        self.model_trainer = ModelTrainer(
            model=config['training']['model'],
            param_grid=config['training'].get('param_grid', None),
            search_type=config['training'].get('search_type', 'grid'),
            scoring=config['training'].get('scoring', 'f1'),
            n_iter=config['training'].get('n_iter', 10),
            save_path=config['training'].get('save_path', 'trained_model.joblib')
        )

        self.model_evaluator = ModelEvaluator(
            average=config['evaluation'].get('average', 'weighted')
        )

    def run(self, X, y):
        """
        Run the full pipeline: data preprocessing, feature engineering, feature selection,
        model training, and evaluation.

        Parameters:
        - X: Features of the dataset.
        - y: Target labels of the dataset.

        Returns:
        - metrics (dict): Evaluation metrics of the final model on the test set.
        """
        # 1. Data Preprocessing
        X_preprocessed, y_preprocessed = self.data_preprocessor.fit_transform(X, y)

        # 2. Feature Engineering
        X_engineered = self.feature_engineer.fit_transform(X_preprocessed)

        # 3. Feature Selection
        X_selected = self.feature_selector.fit_transform(X_engineered, y_preprocessed)

        # 4. Model Training
        best_model, best_params = self.model_trainer.train(X_selected, y_preprocessed)
        print("Best Model Parameters:", best_params)

        # 5. Model Evaluation
        metrics = self.model_evaluator.evaluate(best_model, X_selected, y_preprocessed)
        print("Evaluation Metrics:", metrics)

        return metrics