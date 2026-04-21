from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class preprocess(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, list):
            X = [X]
        processed_dfs = []

        for df in X:
            df = df.copy()
            df.drop(columns=["unit", "source", "code"], inplace=True, errors='ignore')
            df["value"] = df.fillna(0)["value"].astype('int')
            processed_dfs.append(df)

        return pd.concat(processed_dfs, ignore_index=True)


class table_structure(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X_long = X.copy()
        elif isinstance(X, list):
            X_long = pd.concat(X, ignore_index=True)
        else:
            raise ValueError("Input X must be a DataFrame or a list of DataFrames.")

        for col in X_long.columns:
            if pd.api.types.is_object_dtype(X_long[col]):
                X_long[col] = X_long[col].astype(str).str.strip()

        return X_long


class data_split(BaseEstimator, TransformerMixin):
    def fit(self, X_long, y=None):
        return self

    def transform(self, X_long, y=None):
        X_long = X_long.copy()
        X_long[["Province", "Cluster", "Type of Crime"]] = X_long["title"].str.split("—", expand=True)
        X_long["Province"] = X_long["Province"].str.split(":", expand=True)[1]
        X_long["Cluster"] = X_long["Cluster"].str.split(":", expand=True)[1]
        return X_long

class data_cleaning(BaseEstimator, TransformerMixin):
    def fit(self, X_long, y=None):
        return self

    def transform(self, X_long, y=None):
        X_long = X_long.copy()
        X_long["Type of Crime"] = X_long["Type of Crime"].replace(
            ' SRCXR: Robbery: Bank, Street and Cash in transit',
            'SRCXR: Bank Robbery (Street and Cash in transit)')
        X_long["Type of Crime"] = X_long["Type of Crime"].str.split(":", expand=True)[1]
        X_long["Type of Crime"] = X_long["Type of Crime"].str.replace(r'\d+\s*', '', regex=True)
        return X_long


class feature_engineering_crime(BaseEstimator, TransformerMixin):
    def fit(self, X_long, y=None):
        return self

    def transform(self, X_long, y=None):
        X_long = X_long.copy()
        X_long = X_long.drop(columns=["title"], errors='ignore')
        X_long["date"] = pd.to_datetime(X_long["date"])
        X_long = X_long[["date", "Cluster", "Province", "Type of Crime", "value"]]
        return X_long


class feature_engineering_socioeconomic(BaseEstimator, TransformerMixin):
    def fit(self, X_long, y=None):
        return self

    def transform(self, X_long, y=None):
        X_long = X_long.copy()

        # Extract Cluster from title: text between ': ' and ' ('
        # e.g. 'P9D01M1: Greater Giyani (LIM331) — ...' → 'Greater Giyani'
        X_long["Cluster"] = X_long["title"].str.extract(r':\s*([^(]+)\s*\(')
        X_long["Cluster"] = X_long["Cluster"].str.strip()

        X_long = X_long.drop(columns=["title"], errors='ignore')
        X_long["date"] = pd.to_datetime(X_long["date"])
        X_long = X_long[["date", "Cluster", "value"]]
        return X_long


class crime_data_cleaning(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X.rename(columns={"value": "Crime Count"}, inplace=True)
        X.drop(columns=["Year", "Month", "Day", "Province"], inplace=True, errors='ignore')
        X["Cluster"] = X["Cluster"].str.strip()
        X["Cluster"] = X["Cluster"].replace({
            "Bela Bela Cluster": "Bela Bela",
            'Burgersfort Cluster': "Burgersfort",
            'Giyani Cluster': "Giyani",
            'Lephalale Cluster': "Lephalale",
            'Groblersdal Cluster (inc. Laersdrift post-Mar-18)': "Groblersdal",
            'Mokopane Cluster': "Mokopane",
            'Mahwelereng Cluster (inc. part of Moletlane pre-May-21)': "Mahwelereng",
            'Makhado Cluster': "Makhado",
            'Mankweng Cluster': "Polokwane",
            'Modimolle Cluster': "Modimolle",
            'Seshego Cluster': "Polokwane",
            'Thohoyandou Cluster': "Thohoyandou",
            'Tzaneen Cluster': "Tzaneen"
        })
        return X


class socio_data_cleaning(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X["Cluster"] = X["Cluster"].replace({
            'Bela-Bela': "Bela Bela",
            'Fetakgomo-Greater Tubatse': "Burgersfort",
            'Elias Motsoaledi': "Groblersdal",
            'Greater Giyani': "Giyani",
            'Greater Tzaneen': "Tzaneen",
            'Limpopo — Unemployed': np.nan,
            'Modimolle-Mookgophong': "Modimolle",
            'Mogalakwena': "Mahwelereng",
            'Polokwane': "Polokwane",
            'Thulamela': "Thohoyandou"
        })
        X.dropna(subset=["Cluster"], inplace=True)
        return X


class date_filter(BaseEstimator, TransformerMixin):
    """Filter dataset to only include rows from 2006 onwards."""
    def __init__(self, start_year: int = 2006):
        self.start_year = start_year

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X["date"] = pd.to_datetime(X["date"])
        X = X[X["date"].dt.year >= self.start_year]
        X = X.sort_values(by=["date", "Cluster"]).reset_index(drop=True)
        return X


class socio_merger(BaseEstimator, TransformerMixin):
    """Merge a list of DataFrames on date and Cluster."""
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, list):
            raise ValueError("Input must be a list of DataFrames.")

        result = X[0]
        for df in X[1:]:
            result = pd.merge(result, df, on=["date", "Cluster"], how="outer")

        result.sort_values(by=["date", "Cluster"], inplace=True)
        result.dropna(inplace=True)
        result.drop_duplicates(inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result

    # reference: NeuralNine Professional Preprocessing with Pipelines in Python