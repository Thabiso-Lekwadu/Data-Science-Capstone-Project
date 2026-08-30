"""Preprocessing transformers for the crime + socioeconomic datasets.

Each class is a scikit-learn style transformer (``fit`` is a no-op, all the
work is in ``transform``) so the preprocessing steps compose cleanly in the
Kedro nodes via ``.fit_transform(...)`` chaining.

reference: NeuralNine "Professional Preprocessing with Pipelines in Python".
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class preprocess(BaseEstimator, TransformerMixin):
    """Drop metadata columns and coerce ``value`` to a clean non-negative int.

    ``value`` arrives as a float (Quantec standardised figures, e.g.
    4965.21). Previously this used ``df.fillna(0)["value"].astype(int)``,
    which (a) built a throwaway full-frame copy just to fill one column and
    (b) *truncated* rather than rounded (4965.9 -> 4965). We now coerce
    non-numeric junk to NaN, fill with 0, round, and clip at 0 so a crime
    count can never be negative.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, list):
            X = [X]
        processed_dfs = []
        for df in X:
            df = df.copy()
            df.drop(columns=["unit", "source", "code"], inplace=True, errors="ignore")
            value = pd.to_numeric(df["value"], errors="coerce").fillna(0)
            df["value"] = value.round().clip(lower=0).astype("int64")
            processed_dfs.append(df)
        return pd.concat(processed_dfs, ignore_index=True)


class table_structure(BaseEstimator, TransformerMixin):
    """Normalise into a single long table and strip whitespace from strings."""

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
    """Split the crime ``title`` into Province / Cluster / Type of Crime.

    Crime titles are three em-dash-separated segments, e.g.
    ``"P9: Limpopo — LIC01S00: Bela Bela Cluster — T: 25 Serious and Other
    crimes"``. We guard the split so a malformed title (not exactly two
    em-dashes) raises a clear error here instead of a cryptic
    "Columns must be same length as key" from pandas.
    """

    def fit(self, X_long, y=None):
        return self

    def transform(self, X_long, y=None):
        X_long = X_long.copy()
        parts = X_long["title"].str.split("—", expand=True)
        if parts.shape[1] != 3:
            counts = X_long["title"].str.count("—")
            bad = X_long.loc[counts != 2, "title"].drop_duplicates().head(5).tolist()
            raise ValueError(
                "data_split expected every crime title to have exactly two '—' "
                f"separators (Province — Cluster — Type of Crime), but found titles "
                f"with a different structure. Examples: {bad}"
            )
        X_long[["Province", "Cluster", "Type of Crime"]] = parts
        # Each segment is 'CODE: Human readable' — keep the part after the first ':'.
        X_long["Province"] = X_long["Province"].str.split(":", n=1, expand=True)[1]
        X_long["Cluster"] = X_long["Cluster"].str.split(":", n=1, expand=True)[1]
        return X_long


class data_cleaning(BaseEstimator, TransformerMixin):
    """Clean the ``Type of Crime`` label (strip code prefix + leading counter)."""

    def fit(self, X_long, y=None):
        return self

    def transform(self, X_long, y=None):
        X_long = X_long.copy()
        X_long["Type of Crime"] = X_long["Type of Crime"].replace(
            " SRCXR: Robbery: Bank, Street and Cash in transit",
            "SRCXR: Bank Robbery (Street and Cash in transit)",
        )
        # Drop the leading 'CODE:' prefix, then the leading numeric counter
        # ('25 Serious and Other crimes' -> 'Serious and Other crimes').
        X_long["Type of Crime"] = X_long["Type of Crime"].str.split(":", n=1, expand=True)[1]
        X_long["Type of Crime"] = X_long["Type of Crime"].str.replace(r"\d+\s*", "", regex=True)
        # Previously the label kept a leading space (' Murder'), which then leaked
        # into the one-hot column names ('crime_type_ Murder'). Strip it once, here.
        X_long["Type of Crime"] = X_long["Type of Crime"].str.strip()
        return X_long


class feature_engineering_crime(BaseEstimator, TransformerMixin):
    """Keep only the modelling columns and parse the date."""

    def fit(self, X_long, y=None):
        return self

    def transform(self, X_long, y=None):
        X_long = X_long.copy()
        X_long = X_long.drop(columns=["title"], errors="ignore")
        X_long["date"] = pd.to_datetime(X_long["date"])
        X_long = X_long[["date", "Cluster", "Province", "Type of Crime", "value"]]
        return X_long


class feature_engineering_socioeconomic(BaseEstimator, TransformerMixin):
    """Extract the cluster name from a socioeconomic ``title`` and keep date/value."""

    def fit(self, X_long, y=None):
        return self

    def transform(self, X_long, y=None):
        X_long = X_long.copy()
        # Extract Cluster from title: text between ': ' and ' ('
        # e.g. 'P9D01M1: Greater Giyani (LIM331) — ...' → 'Greater Giyani'
        X_long["Cluster"] = X_long["title"].str.extract(r":\s*([^(]+)\s*\(")
        X_long["Cluster"] = X_long["Cluster"].str.strip()
        X_long = X_long.drop(columns=["title"], errors="ignore")
        X_long["date"] = pd.to_datetime(X_long["date"])
        X_long = X_long[["date", "Cluster", "value"]]
        return X_long


class crime_data_cleaning(BaseEstimator, TransformerMixin):
    """Rename SAPS police clusters to the geographic cluster names used by the
    socioeconomic data, so the two sources can be merged on a common key.

    NOTE: two police clusters ('Mankweng Cluster' and 'Seshego Cluster') both
    map onto the 'Polokwane' municipality. This *intentionally* produces two
    rows per (date, Type of Crime) for Polokwane; they are summed back into a
    single Polokwane total by ``aggregate_crime_duplicates`` immediately
    after this step (see the node). Do NOT rely on uniqueness of
    (date, Cluster, Type of Crime) until that aggregation has run.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X.rename(columns={"value": "Crime Count"}, inplace=True)
        X.drop(columns=["Year", "Month", "Day", "Province"], inplace=True, errors="ignore")
        X["Cluster"] = X["Cluster"].str.strip()
        X["Cluster"] = X["Cluster"].replace({
            "Bela Bela Cluster": "Bela Bela",
            "Burgersfort Cluster": "Burgersfort",
            "Giyani Cluster": "Giyani",
            "Lephalale Cluster": "Lephalale",
            "Groblersdal Cluster (inc. Laersdrift post-Mar-18)": "Groblersdal",
            "Mokopane Cluster": "Mokopane",
            "Mahwelereng Cluster (inc. part of Moletlane pre-May-21)": "Mahwelereng",
            "Makhado Cluster": "Makhado",
            "Mankweng Cluster": "Polokwane",
            "Modimolle Cluster": "Modimolle",
            "Seshego Cluster": "Polokwane",
            "Thohoyandou Cluster": "Thohoyandou",
            "Tzaneen Cluster": "Tzaneen",
        })
        return X


class exclude_crime_categories(BaseEstimator, TransformerMixin):
    """Drop SAPS roll-up/aggregate categories and deprecated 'Old *' categories.

    The raw SAPS feed mixes *granular* crime types (Murder, Rape, Burglary…)
    with *aggregate* roll-ups that are themselves sums of those granular rows
    ('Contact crimes (Crimes against the person)', 'Property-related crimes',
    the grand total 'Serious and Other crimes', etc.). Modelling 'Crime Count'
    across a target column that mixes granular counts and their own subtotals
    is a data-integrity problem: the same incidents are represented at two
    different levels of aggregation.

    The excluded set is configuration, not hard-coded policy — it comes from
    ``parameters.yml: excluded_crime_types`` so it can be reviewed/adjusted
    against the SAPS taxonomy. Matching is on the whitespace-stripped label.
    """

    def __init__(self, excluded: Optional[Iterable[str]] = None):
        self.excluded = list(excluded) if excluded else []

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if not self.excluded:
            return X
        excluded = {e.strip() for e in self.excluded}
        mask = X["Type of Crime"].str.strip().isin(excluded)
        return X.loc[~mask].reset_index(drop=True)


class aggregate_crime_duplicates(BaseEstimator, TransformerMixin):
    """Collapse duplicate (date, Cluster, Type of Crime) rows by summing.

    This is what fixes the 'Polokwane' double-count: after
    ``crime_data_cleaning`` maps both the Mankweng and Seshego police clusters
    onto 'Polokwane', Polokwane has two rows per (date, Type of Crime) — one
    from each source cluster — that were never combined. Left unsummed they
    corrupt (a) the regression target and (b) the per-(Cluster, Type of Crime)
    lag/rolling features built downstream, whose ``groupby.shift`` silently
    interleaves the two series. Summing yields a single, correct Polokwane
    total per crime type per period.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        return (
            X.groupby(["date", "Cluster", "Type of Crime"], as_index=False)["Crime Count"]
            .sum()
        )


class socio_data_cleaning(BaseEstimator, TransformerMixin):
    """Rename socioeconomic cluster names to the common geographic key and drop
    non-mappable ('Limpopo — Unemployed' province-level) rows."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X["Cluster"] = X["Cluster"].replace({
            "Bela-Bela": "Bela Bela",
            "Fetakgomo-Greater Tubatse": "Burgersfort",
            "Elias Motsoaledi": "Groblersdal",
            "Greater Giyani": "Giyani",
            "Greater Tzaneen": "Tzaneen",
            "Limpopo — Unemployed": np.nan,
            "Modimolle-Mookgophong": "Modimolle",
            "Mogalakwena": "Mahwelereng",
            "Polokwane": "Polokwane",
            "Thulamela": "Thohoyandou",
        })
        X.dropna(subset=["Cluster"], inplace=True)
        return X


class date_filter(BaseEstimator, TransformerMixin):
    """Filter dataset to only include rows from ``start_year`` onwards."""

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
    """Merge a list of DataFrames on date and Cluster.

    Every dataframe merged in (the ``df`` on the right of each join) is
    expected to contribute exactly one row per (date, Cluster) -- that is
    the whole point of a socioeconomic indicator: one value per
    cluster-period. If some upstream dataset instead has multiple rows per
    (date, Cluster) -- e.g. several unaggregated categories, like the 6
    separate pop_edu_N sources -- a plain merge silently turns into a
    many-to-many join and the row count explodes multiplicatively at every
    step (this has bitten this pipeline before).

    ``validate="many_to_one"`` turns that silent explosion into an immediate,
    specific pandas MergeError naming the duplicate keys, so the broken
    input is obvious instead of surfacing 10+ steps later as an
    "ExcelDataset sheet too large" error.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, list):
            raise ValueError("Input must be a list of DataFrames.")

        result = X[0]
        for i, df in enumerate(X[1:], start=1):
            try:
                result = pd.merge(
                    result, df, on=["date", "Cluster"], how="outer", validate="many_to_one"
                )
            except pd.errors.MergeError as e:
                dupes = df[df.duplicated(subset=["date", "Cluster"], keep=False)]
                dupes = dupes.sort_values(["date", "Cluster"])
                n_pairs = dupes[["date", "Cluster"]].drop_duplicates().shape[0]
                raise ValueError(
                    f"socio_merger: input at position {i} has duplicate (date, Cluster) "
                    f"keys -- merging it would silently multiply row counts. "
                    f"{len(dupes)} rows involved across {n_pairs} duplicated (date, Cluster) "
                    f"pairs. This dataset needs to be aggregated "
                    f"(e.g. groupby(['date', 'Cluster']).sum()/.mean()) to one row per "
                    f"cluster-period before merging. Sample duplicates:\n"
                    f"{dupes.head(10).to_string(index=False)}"
                ) from e

        result.sort_values(by=["date", "Cluster"], inplace=True)
        result.dropna(inplace=True)
        result.drop_duplicates(inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result
