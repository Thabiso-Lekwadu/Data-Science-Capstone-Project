# extend the scikit learn base estimator to make use of the fit method
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config
import pandas as pd

set_config(transform_output="pandas")

class preprocess(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X[4:]
        X = X.drop(range(5, 8))
        X.columns = X.iloc[0]
        X = X.iloc[1:]
        X = X.rename(columns={'title': 'Date'})
        X = X.set_index("Date")
        X.index = pd.to_datetime(X.index)

        return X

class imput_missing(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Convert all columns to numeric, turning invalid strings to NaN
        X = X.fillna(0).round(0).astype(int)
        return X


class table_structure(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transposed = X.T.reset_index().rename(columns={4: 'Region'})
        id_vars = X_transposed.columns[0]
        date_cols = X_transposed.columns[1:]
        X_long = pd.melt(X_transposed, id_vars=id_vars, value_vars=date_cols, var_name='Date', value_name='Value')
        return X_long

class data_split(BaseEstimator, TransformerMixin):
    def fit(self, X_long, y=None):
        return self

    def transform(self, X_long, y=None):
        X_long[["Province", "Cluster", "Type of Crime"]] = X_long["Region"].str.split("—", expand=True)
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
        X_long.drop(columns="Region", inplace=True)
        X_long["Date"] = pd.to_datetime(X_long["Date"])
        X_long["Year"] = X_long["Date"].dt.year
        X_long["Month"] = X_long["Date"].dt.month_name()
        X_long["Day"] = X_long["Date"].dt.day_name()
        X_long = X_long[["Date", "Year", "Month", "Day", "Province", "Cluster", "Type of Crime", "Value"]]
        return X_long

    # reference: NeuralNine Professional Preprocessing with Pipelines in Python