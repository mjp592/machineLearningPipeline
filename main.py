from src.feature_type_detector import FeatureTypeDetector
from src.data_preprocessing import DataPreprocessor
import pandas as pd

if __name__ == '__main__':
    #load sample data
    df = pd.read_csv('data/sample-data.csv')
    target_df = df['target']
    feature_df = df.drop('target', axis=1)

    #initialize feature type detector
    feature_detector = FeatureTypeDetector()

    #detect feature types
    feature_types = feature_detector.detect_feature_types(feature_df)

    #unpack feature types
    continuous_features = feature_types['continuous']
    categorical_features = feature_types['categorical']
    binary_features = feature_types['binary']
    ordinal_features = feature_types['ordinal']

    print(feature_types)

    #initialize data preprocessor
    data_preprocessor = DataPreprocessor(continuous_features, categorical_features,
                                         binary_features, ordinal_features,
                                         ordinal_order=['low', 'medium', 'high'],
                                         non_negative_features=['continuous_feature'],
                                         balance_method='oversample')

    #preprocess data
    data_preprocessor.fit(feature_df)
    transformed_df, target_df = data_preprocessor.transform(feature_df, target_df)

    # print transformed data
    print(feature_df)
    print(transformed_df)
    print(target_df)
