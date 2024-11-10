from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.decomposition import PCA
import pandas as pd
import itertools


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, interaction_features=None, polynomial_degree=None, binning_features=None, n_bins=5,
                 binning_strategy="uniform", pca_components=None):
        """
        Initialize with various options for feature engineering.

        Parameters:
        interaction_features (list): List of feature names to consider for pairwise interactions.
        polynomial_degree (int): Degree of polynomial features to generate (e.g., 2 for quadratic).
        binning_features (list): List of features to bin.
        n_bins (int): Number of bins to create for binning features.
        binning_strategy (str): Strategy for binning ('uniform', 'quantile', 'kmeans').
        pca_components (int): Number of PCA components to retain.
        """
        self.interaction_features = interaction_features
        self.polynomial_degree = polynomial_degree
        self.binning_features = binning_features
        self.n_bins = n_bins
        self.binning_strategy = binning_strategy
        self.pca_components = pca_components
        self.interaction_terms_ = []  # Store names of generated interaction terms

    def fit(self, X, y=None):
        # Fitting components for polynomial and PCA if specified
        if self.polynomial_degree:
            self.poly = PolynomialFeatures(degree=self.polynomial_degree, include_bias=False)
            self.poly.fit(X[self.interaction_features])  # Fit on specified interaction features
        if self.pca_components:
            self.pca = PCA(n_components=self.pca_components)
            self.pca.fit(X)
        if self.binning_features:
            self.binner = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy=self.binning_strategy)
            self.binner.fit(X[self.binning_features])

        return self

    def transform(self, X):
        X_transformed = X.copy()

        # 1. Generate interaction features if specified
        if self.interaction_features:
            for feature1, feature2 in itertools.combinations(self.interaction_features, 2):
                interaction_name = f"{feature1}_x_{feature2}"
                X_transformed[interaction_name] = X_transformed[feature1] * X_transformed[feature2]
                self.interaction_terms_.append(interaction_name)

        # 2. Generate polynomial features if specified
        if self.polynomial_degree:
            poly_features = self.poly.transform(X[self.interaction_features])
            poly_feature_names = self.poly.get_feature_names_out(self.interaction_features)
            poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=X.index)
            X_transformed = pd.concat([X_transformed, poly_df], axis=1)

        # 3. Apply binning to specified features if specified
        if self.binning_features:
            binned_features = self.binner.transform(X[self.binning_features])
            binned_feature_names = [f"{feature}_binned" for feature in self.binning_features]
            binned_df = pd.DataFrame(binned_features, columns=binned_feature_names, index=X.index)
            X_transformed = pd.concat([X_transformed, binned_df], axis=1)

        # 4. Apply PCA if specified
        if self.pca_components:
            pca_features = self.pca.transform(X)
            pca_feature_names = [f"pca_{i + 1}" for i in range(self.pca_components)]
            pca_df = pd.DataFrame(pca_features, columns=pca_feature_names, index=X.index)
            X_transformed = pd.concat([X_transformed, pca_df], axis=1)

        return X_transformed
