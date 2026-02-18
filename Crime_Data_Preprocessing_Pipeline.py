# extend the scikit learn base estimator to make use of the fit method
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config
import pandas as pd

set_config(transform_output="pandas")

class preprocess(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, list):
            X = [X]
        processed_dfs = []

        for df in X:
            # Create a copy to avoid modifying original data unexpectedly
            # Your original cleaning logic applied to each individual dataframe
            df.drop(columns=["unit", "source", "code"], inplace=True, errors='ignore')
            df["value"] = df.fillna(0).value.astype('int')
            df = df.rename(columns={'title': 'Date'})
            df = df.set_index("Date")

            processed_dfs.append(df)

        # Combine all processed dataframes into one single dataset
        return pd.concat(processed_dfs)


class table_structure(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Assumes X is a list of DataFrames: [df1, df2, ..., dfN]
        if not isinstance(X, list):
            raise ValueError("Input X must be a list of DataFrames.")

        X_long = pd.concat(X, ignore_index=True)
        for col in X_long.select_dtypes(include="object").columns:
            X_long[col] = X_long[col].str.strip()

        return X_long

class data_split(BaseEstimator, TransformerMixin):
    def fit(self, X_long, y=None):
        return self

    def transform(self, X_long, y=None):
        X_long[["Province", "Cluster", "Type of Crime"]] = X_long["title"].str.split("—", expand=True)
        X_long["Province"] = X_long["Province"].str.split(":", expand=True).drop(columns=0)
        X_long["Cluster"] = X_long["Cluster"].str.split(":", expand=True).drop(columns=0)
        return X_long

class data_cleaning(BaseEstimator, TransformerMixin):
    def fit(self, X_long, y=None):
        return self

    def transform(self, X_long, y=None):
        X_long["Type of Crime"] = X_long["Type of Crime"].replace(
            ' SRCXR: Robbery: Bank, Street and Cash in transit', 'SRCXR: Bank Robbery (Street and Cash in transit)')
        X_long["Type of Crime"] = X_long["Type of Crime"].str.split(":", expand=True).drop(columns=0)
        X_long["Type of Crime"] = X_long["Type of Crime"].str.replace(r'\d+\s*','', regex=True)
        return X_long

class feature_engineering(BaseEstimator, TransformerMixin):
    def fit(self, X_long, y=None):
        return self

    def transform(self, X_long, y=None):
        X_long.drop(columns="title", inplace=True)
        X_long["date"] = pd.to_datetime(X_long["date"])
        X_long["Year"] = X_long["date"].dt.year
        X_long["Month"] = X_long["date"].dt.month_name()
        X_long["Day"] = X_long["date"].dt.day_name()
        X_long = X_long[["date", "Year", "Month", "Day", "Province", "Cluster", "Type of Crime", "value"]]
        return X_long

    # reference: NeuralNine Professional Preprocessing with Pipelines in Python