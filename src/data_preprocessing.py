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
                 ordinal_order, non_negative_features, outlier_threshold=3,
                 skew_threshold=0.5, missing_threshold=0.1, balance_method=None):
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
        self.non_negative_features = non_negative_features
        self.outlier_threshold = outlier_threshold
        self.skew_threshold = skew_threshold
        self.missing_threshold = missing_threshold
        self.balance_method = balance_method
        self.processor = None

    def drop_high_missing_columns(self, df):
        """
        Drop columns that have more than the specified percentage of missing values.

        Parameters:
        - df (pd.DataFrame): Input DataFrame.
        - missing_threshold (float): Percentage threshold (between 0 and 1) for dropping columns.

        Returns:
        - pd.DataFrame: DataFrame with columns dropped based on the missing threshold.
        """
        # Calculate the percentage of missing values for each column
        missing_percentages = df.isnull().mean()

        # Identify columns to drop
        columns_to_drop = missing_percentages[missing_percentages > self.missing_threshold].index
        print(f"Dropping columns with more than {self.missing_threshold * 100}% missing values: {list(columns_to_drop)}")

        # Drop columns
        df_dropped = df.drop(columns=columns_to_drop, axis=1)


        for col in columns_to_drop:
            try:
                self.continuous_features.remove(col)
            except ValueError:
                pass

            try:
                self.non_negative_features.remove(col)
            except ValueError:
                pass

            try:
                self.categorical_features.remove(col)
            except ValueError:
                pass

            try:
                self.ordinal_features.remove(col)
            except ValueError:
                pass

            try:
                self.binary_features.remove(col)
            except ValueError:
                pass

        return df_dropped

    def detect_and_remove_errors(self, x):
        """
        Detect and remove data errors such as outliers, invalid entries, and data type mismatches.
        """
        x_cleaned = x.copy()

        # 1. Detect and remove outliers for continuous features
        for feature in self.continuous_features:
            # Calculate Z-scores
            z_scores = (x_cleaned[feature] - x_cleaned[feature].mean()) / x_cleaned[feature].std()
            z_scores.fillna(0, inplace=True)
            x_cleaned = x_cleaned[(np.abs(z_scores) < self.outlier_threshold)]

        # 2. Detect invalid entries (e.g., negative values where they shouldn't be)
        for feature in self.non_negative_features:
            if (x_cleaned[feature] < 0).any():  # Assuming negative values are invalid
                x_cleaned = x_cleaned[x_cleaned[feature] >= 0]

        # 3. Verify and enforce data types
        # Continuous features should be numeric
        for feature in self.continuous_features:
            if not np.issubdtype(x_cleaned[feature].dtype, np.number):
                x_cleaned[feature] = pd.to_numeric(x_cleaned[feature], errors='coerce')

        # 4. Drop columns with more than the specified percentage of missing values
        x_cleaned = self.drop_high_missing_columns(x_cleaned)

        return x_cleaned

    def apply_log_transformation(self, x):
        """
        Apply log transformation to skewed continuous features based on skew_threshold.
        """
        x_transformed = x.copy()

        for feature in self.continuous_features:
            # Check skewness
            skewness = x_transformed[feature].skew()
            if abs(skewness) > self.skew_threshold:
                # Apply log transformation and add a small constant to avoid log(0)
                x_transformed[feature] = np.log1p(x_transformed[feature])

        return x_transformed

    def build_preprocessing_pipeline(self):
        # Pipelines for different types of data with missing indicators
        continuous_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=False)),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent', add_indicator=False)),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        binary_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent', add_indicator=False))
        ])

        ordinal_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent', add_indicator=False)),
            ('encoder', OrdinalEncoder(categories=[self.ordinal_order],
                                               handle_unknown='use_encoded_value',
                                               unknown_value=np.nan))
        ])

        # Combine pipelines
        preprocessor = ColumnTransformer([
            ('continuous', continuous_pipeline, self.continuous_features),
            ('categorical', categorical_pipeline, self.categorical_features),
            ('binary', binary_pipeline, self.binary_features),
            ('ordinal', ordinal_pipeline, self.ordinal_features)
        ], verbose=True)

        return preprocessor

    def balance_classes(self, x, y):
        """
        Balance classes in the target variable using specified method (oversample or undersample).
        """
        if self.balance_method == 'oversample':
            smote = SMOTE(random_state=42)
            x_balanced, y_balanced = smote.fit_resample(x, y)
        elif self.balance_method == 'undersample':
            undersampler = RandomUnderSampler()
            x_balanced, y_balanced = undersampler.fit_resample(x, y)
        else:
            # If no balancing method is specified, return original X and y
            x_balanced, y_balanced = x, y
        return x_balanced, y_balanced

    def fit(self, x):
        # Clean data, apply log transformation, then build + fit the preprocessing pipeline
        x_cleaned = self.detect_and_remove_errors(x)
        x_transformed = self.apply_log_transformation(x_cleaned)

        self.processor = self.build_preprocessing_pipeline()
        self.processor.fit(x_transformed)
        return self

    def transform(self, x, y=None):
        # Clean data, apply log transformation, then transform with the fitted pipeline
        x_cleaned = self.detect_and_remove_errors(x)
        x_transformed = self.apply_log_transformation(x_cleaned)
        x_final = self.processor.transform(x_transformed)

        # Balance classes after transformations, if balance method is specified and y is provided
        if y is not None and self.balance_method:
            x_final, y = self.balance_classes(x_final, y)

        # Get categorical feature names for final dataframe
        try:
            cat_encoder =self.processor.named_transformers_['categorical']['encoder']
            cat_feature_names = cat_encoder.categories_[0]
        except AttributeError:
            cat_feature_names = []

        # Create final feature name list in order of pipeline
        feature_names = []
        feature_names.extend(self.continuous_features)
        feature_names.extend(cat_feature_names)
        feature_names.extend(self.binary_features)
        feature_names.extend(self.ordinal_features)

        # Create final dataframe
        final_feature_df = pd.DataFrame(x_final, columns=feature_names)

        return (final_feature_df, y) if y is not None else final_feature_df
