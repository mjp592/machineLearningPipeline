from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from feature_selection import FeatureSelector
from joblib import load


class InferencePipeline:
    def __init__(self, config, model_path='trained_model.joblib'):
        """
        Initialize the inference pipeline with configuration for preprocessing, feature engineering, and selection.

        Parameters:
        - config (dict): Configuration dictionary with parameters for each stage of the pipeline.
        - model_path (str): Path to the saved trained model.
        """
        # Initialize preprocessing, feature engineering, and feature selection based on config
        self.data_preprocessor = DataPreprocessor(
            continuous_features=config['preprocessing']['continuous_features'],
            categorical_features=config['preprocessing']['categorical_features'],
            binary_features=config['preprocessing']['binary_features'],
            ordinal_features=config['preprocessing']['ordinal_features'],
            ordinal_order=config['preprocessing']['ordinal_order'],
            outlier_threshold=config['preprocessing'].get('outlier_threshold', 3),
            skew_threshold=config['preprocessing'].get('skew_threshold', 0.5)
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

        # Load the trained model
        self.model = load(model_path)

    def predict(self, X):
        """
        Make predictions on new data using the trained model and pipeline transformations.

        Parameters:
        - X: New data for prediction.

        Returns:
        - predictions: Predicted labels.
        """
        # Apply the same preprocessing and feature engineering steps to new data
        X_preprocessed = self.data_preprocessor.transform(X)
        X_engineered = self.feature_engineer.transform(X_preprocessed)
        X_selected = self.feature_selector.transform(X_engineered)

        # Generate predictions
        predictions = self.model.predict(X_selected)
        return predictions

    def predict_proba(self, X):
        """
        Predict class probabilities on new data (if supported by the model).

        Parameters:
        - X: New data for prediction.

        Returns:
        - probabilities: Predicted probabilities for each class.
        """
        X_preprocessed = self.data_preprocessor.transform(X)
        X_engineered = self.feature_engineer.transform(X_preprocessed)
        X_selected = self.feature_selector.transform(X_engineered)

        # Generate predicted probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_selected)
            return probabilities
        else:
            raise AttributeError("The loaded model does not support probability predictions.")
