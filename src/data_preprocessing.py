from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, continuous_features, categorical_features, binary_features, ordinal_features,
                 ordinal_order, outlier_threshold=3, skew_threshold=0.5, balance_method=None):
        """
        Initializes the data preprocessor with options for cleaning, transformation, and balancing.

        Parameters:
        - continuous_features (list): List of continuous feature names.
        - categorical_features (list): List of categorical feature names.
        - binary_features (list): List of binary feature names.
        - ordinal_features (list): List of ordinal feature names.
        - ordinal_order (list): List of order values for ordinal features.
        - outlier_threshold (float): Z-score threshold for outlier detection.
        - skew_threshold (float): Skewness threshold for log transformation.
        - balance_method (str): Method for class balancing, 'oversample' for SMOTE, 'undersample' for random undersampling.
        """
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.binary_features = binary_features
        self.ordinal_features = ordinal_features
        self.ordinal_order = ordinal_order
        self.outlier_threshold = outlier_threshold
        self.skew_threshold = skew_threshold
        self.balance_method = balance_method
        self.selected_features_ = None

    def detect_and_remove_errors(self, X):
        """
        Detect and remove data errors such as outliers, invalid entries, and data type mismatches.
        """
        X_cleaned = X.copy()

        # 1. Detect and remove outliers for continuous features
        for feature in self.continuous_features:
            # Calculate Z-scores
            z_scores = (X_cleaned[feature] - X_cleaned[feature].mean()) / X_cleaned[feature].std()
            X_cleaned = X_cleaned[(np.abs(z_scores) < self.outlier_threshold)]

        # 2. Detect invalid entries (e.g., negative values where they shouldn't be)
        for feature in self.continuous_features:
            if (X_cleaned[feature] < 0).any():  # Assuming negative values are invalid
                X_cleaned = X_cleaned[X_cleaned[feature] >= 0]

        # 3. Verify and enforce data types
        # Continuous features should be numeric
        for feature in self.continuous_features:
            if not np.issubdtype(X_cleaned[feature].dtype, np.number):
                X_cleaned[feature] = pd.to_numeric(X_cleaned[feature], errors='coerce')

        # Drop any rows with remaining NaN values from type coercion
        X_cleaned.dropna(subset=self.continuous_features, inplace=True)

        return X_cleaned

    def apply_log_transformation(self, X):
        """
        Apply log transformation to skewed continuous features based on skew_threshold.
        """
        X_transformed = X.copy()

        for feature in self.continuous_features:
            # Check skewness
            skewness = X_transformed[feature].skew()
            if abs(skewness) > self.skew_threshold:
                # Apply log transformation and add a small constant to avoid log(0)
                X_transformed[feature] = np.log1p(X_transformed[feature])

        return X_transformed

    def build_preprocessing_pipeline(self):
        # Pipelines for different types of data with missing indicators
        continuous_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True)),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        binary_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True))
        ])

        ordinal_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True)),
            ('ordinal_encoder', OrdinalEncoder(categories=[self.ordinal_order]))
        ])

        # Combine pipelines
        preprocessor = ColumnTransformer([
            ('continuous', continuous_pipeline, self.continuous_features),
            ('categorical', categorical_pipeline, self.categorical_features),
            ('binary', binary_pipeline, self.binary_features),
            ('ordinal', ordinal_pipeline, self.ordinal_features)
        ])

        return preprocessor

    def balance_classes(self, X, y):
        """
        Balance classes in the target variable using specified method (oversample or undersample).
        """
        if self.balance_method == 'oversample':
            smote = SMOTE()
            X_balanced, y_balanced = smote.fit_resample(X, y)
        elif self.balance_method == 'undersample':
            undersampler = RandomUnderSampler()
            X_balanced, y_balanced = undersampler.fit_resample(X, y)
        else:
            # If no balancing method is specified, return original X and y
            X_balanced, y_balanced = X, y
        return X_balanced, y_balanced

    def fit(self, X, y=None):
        # Clean data, apply log transformation, then fit the preprocessing pipeline
        X_cleaned = self.detect_and_remove_errors(X)
        X_transformed = self.apply_log_transformation(X_cleaned)

        # Balance classes after cleaning but before fitting the pipeline
        if y is not None and self.balance_method:
            X_transformed, y = self.balance_classes(X_transformed, y)

        self.preprocessor = self.build_preprocessing_pipeline()
        self.preprocessor.fit(X_transformed)
        return self

    def transform(self, X, y=None):
        # Clean data, apply log transformation, then transform with the fitted pipeline
        X_cleaned = self.detect_and_remove_errors(X)
        X_transformed = self.apply_log_transformation(X_cleaned)

        # Balance classes after transformations, if balance method is specified and y is provided
        if y is not None and self.balance_method:
            X_transformed, y = self.balance_classes(X_transformed, y)

        X_final = self.preprocessor.transform(X_transformed)
        return (X_final, y) if y is not None else X_final
