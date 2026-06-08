# src/preprocessing.py

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def split_features(X):

    # Automatically separate numerical and categorical features.
    numeric_features = (
        X.select_dtypes(include=["number"])
        .columns
        .tolist()
    )

    categorical_features = (
        X.select_dtypes(exclude=["number"])
        .columns
        .tolist()
    )

    return numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):

    # Build preprocessing pipeline.
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    return preprocessor