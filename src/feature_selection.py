from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFE
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, variance_threshold=None, correlation_threshold=None,
                 use_rfe=False, rfe_model=None, use_lasso=False, lasso_alpha=1.0,
                 use_tree_importance=False, tree_model=None, n_features=None):
        """
        Initialize the feature selector with various feature selection techniques.

        Parameters:
        - variance_threshold (float): Threshold for variance-based filtering.
        - correlation_threshold (float): Threshold for correlation-based filtering.
        - use_rfe (bool): Whether to use Recursive Feature Elimination.
        - rfe_model: Model to use for RFE (e.g., RandomForest, LinearRegression).
        - use_lasso (bool): Whether to use Lasso regularization for feature selection.
        - lasso_alpha (float): Alpha parameter for Lasso.
        - use_tree_importance (bool): Whether to use tree-based feature importance.
        - tree_model: Model to use for tree-based feature selection (e.g., RandomForest).
        - n_features (int): Number of features to select (used by RFE and tree-based feature importance).
        """
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.use_rfe = use_rfe
        self.rfe_model = rfe_model
        self.use_lasso = use_lasso
        self.lasso_alpha = lasso_alpha
        self.use_tree_importance = use_tree_importance
        self.tree_model = tree_model
        self.n_features = n_features
        self.selected_features_ = None

    def fit(self, x, y=None):
        x_transformed = x.copy()

        # 1. Variance Thresholding
        if self.variance_threshold is not None:
            selector = VarianceThreshold(threshold=self.variance_threshold)
            selector.fit(x)
            self.selected_features_ = x.columns[selector.get_support()]
            x_transformed = x_transformed.loc[:, self.selected_features_]

        # 2. Correlation Thresholding
        if self.correlation_threshold is not None:
            corr_matrix = x.corr().abs()
            upper_triangle = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(bool))
            to_drop = [column for column in upper_triangle.columns if
                       any(upper_triangle[column] > self.correlation_threshold)]
            x_transformed = x_transformed.drop(columns=to_drop)
            self.selected_features_ = x_transformed.columns

        # 3. Recursive Feature Elimination (RFE)
        if self.use_rfe and self.rfe_model is not None:
            rfe = RFE(estimator=self.rfe_model, n_features_to_select=self.n_features)
            rfe.fit(x_transformed, y)
            self.selected_features_ = x_transformed.columns[rfe.support_]
            x_transformed = x_transformed.loc[:, self.selected_features_]

        # 4. Lasso Regularization
        if self.use_lasso:
            lasso = Lasso(alpha=self.lasso_alpha)
            lasso.fit(x_transformed, y)
            lasso_selector = SelectFromModel(lasso, prefit=True)
            self.selected_features_ = x_transformed.columns[lasso_selector.get_support()]
            x_transformed = x_transformed.loc[:, self.selected_features_]

        # 5. Tree-Based Feature Importance
        if self.use_tree_importance and self.tree_model is not None:
            self.tree_model.fit(x_transformed, y)
            importances = self.tree_model.feature_importances_
            importance_df = pd.DataFrame({'feature': x_transformed.columns, 'importance': importances})
            top_features = importance_df.nlargest(self.n_features, 'importance')['feature']
            self.selected_features_ = top_features

        return self

    def transform(self, x):
        # Apply the selected features to the input data
        if self.selected_features_ is not None:
            return x.loc[:, self.selected_features_]
        return x
