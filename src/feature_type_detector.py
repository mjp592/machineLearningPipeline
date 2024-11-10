import pandas as pd
import re


class FeatureTypeDetector:
    def __init__(self, ordinal_threshold=5, known_ordinal_keywords=None):
        """
        Initialize the feature type detector with additional ordinal detection options.

        Parameters:
        - ordinal_threshold (int): The maximum number of unique values to consider a feature as ordinal.
        - known_ordinal_keywords (list): A list of keywords or patterns indicating ordinal order (e.g., ['low', 'medium', 'high']).
        """
        self.ordinal_threshold = ordinal_threshold
        self.known_ordinal_keywords = known_ordinal_keywords or ['low', 'medium', 'high', 'poor', 'fair', 'good',
                                                                 'excellent']
    @staticmethod
    def is_numeric_like(series):
        """Check if all values in a series are numeric-like (e.g., '1', '2', '3')."""
        try:
            series.astype(float)
            return True
        except ValueError:
            return False

    def contains_ordinal_keywords(self, series):
        """Check if a series contains known ordinal keywords."""
        pattern = '|'.join(self.known_ordinal_keywords)
        return any(re.search(pattern, str(value).lower()) for value in series.unique())

    def detect_feature_types(self, df):
        """
        Detect and categorize features in the dataset as continuous, categorical, binary, or ordinal.

        Parameters:
        - df (pd.DataFrame): The dataset to analyze.

        Returns:
        - feature_types (dict): A dictionary containing lists of feature names for each type.
        """
        feature_types = {
            'continuous': [],
            'categorical': [],
            'binary': [],
            'ordinal': []
        }

        for col in df.columns:
            unique_values = df[col].nunique()
            is_numeric = pd.api.types.is_numeric_dtype(df[col])

            # Detect binary features (only two unique values)
            if unique_values == 2:
                feature_types['binary'].append(col)

            # Detect continuous features (numeric with many unique values)
            elif is_numeric and unique_values > self.ordinal_threshold:
                feature_types['continuous'].append(col)

            # Detect ordinal features (numeric-like or contains ordinal keywords)
            elif (self.is_numeric_like(series=df[col]) or self.contains_ordinal_keywords(
                    df[col])) and unique_values <= self.ordinal_threshold:
                feature_types['ordinal'].append(col)

            # Default to categorical if it doesn’t meet other criteria
            else:
                feature_types['categorical'].append(col)

        return feature_types
