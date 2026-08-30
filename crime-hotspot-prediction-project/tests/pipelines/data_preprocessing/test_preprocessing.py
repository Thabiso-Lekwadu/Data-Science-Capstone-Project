"""Regression tests for the crime-data preprocessing fixes.

These guard three concrete bugs that were fixed:
  1. Polokwane double-count: 'Mankweng' and 'Seshego' both map to 'Polokwane'
     but their counts were never summed -> duplicate (date, Cluster, Type)
     rows. `aggregate_crime_duplicates` must collapse them.
  2. SAPS aggregate/roll-up categories (e.g. 'Property-related crimes') were
     mixed into the same target as the granular crimes they sum.
     `exclude_crime_categories` must drop them.
  3. Crime-type labels carried a leading space (' Murder') that leaked into
     one-hot column names.
"""
from pathlib import Path

import pandas as pd
import pytest

from crime_hotspot_prediction_project.pipelines.data_preprocessing.nodes import (
    preprocess_crime_data,
)
from crime_hotspot_prediction_project.pipelines.data_preprocessing.transformers import (
    aggregate_crime_duplicates,
    exclude_crime_categories,
)

RAW = Path(__file__).resolve().parents[3] / "data" / "01_raw" / "crime"

AGGREGATES = [
    "Serious and Other crimes",
    "Property-related crimes",
    "Contact crimes (Crimes against the person)",
    "Old Rape and Attempted Rape",
]


@pytest.fixture(scope="module")
def crime_processed():
    if not RAW.exists():
        pytest.skip("raw crime data not available in this environment")
    dfs = [pd.read_excel(RAW / f"crime_df{i}.xlsx") for i in (1, 2, 3)]
    return preprocess_crime_data(dfs[0], dfs[1], dfs[2], AGGREGATES)


def test_no_duplicate_keys(crime_processed):
    dupes = crime_processed.duplicated(["date", "Cluster", "Type of Crime"]).sum()
    assert dupes == 0, f"{dupes} duplicate (date, Cluster, Type of Crime) rows remain"


def test_aggregates_excluded(crime_processed):
    present = set(crime_processed["Type of Crime"].str.strip())
    leaked = present.intersection(AGGREGATES)
    assert not leaked, f"aggregate categories not excluded: {leaked}"


def test_labels_stripped(crime_processed):
    assert not crime_processed["Type of Crime"].str.startswith(" ").any()
    assert not crime_processed["Type of Crime"].str.endswith(" ").any()


def test_counts_non_negative(crime_processed):
    assert (crime_processed["Crime Count"] >= 0).all()


def test_aggregate_duplicates_sums():
    """Unit test the dedup transformer in isolation: two rows for the same
    (date, Cluster, Type) must sum."""
    df = pd.DataFrame({
        "date": ["2015-12-31", "2015-12-31", "2015-12-31"],
        "Cluster": ["Polokwane", "Polokwane", "Giyani"],
        "Type of Crime": ["Murder", "Murder", "Murder"],
        "Crime Count": [50, 77, 10],
    })
    out = aggregate_crime_duplicates().fit_transform(df)
    pol = out[(out["Cluster"] == "Polokwane") & (out["Type of Crime"] == "Murder")]
    assert len(pol) == 1
    assert int(pol["Crime Count"].iloc[0]) == 127


def test_exclude_is_configurable():
    df = pd.DataFrame({"Type of Crime": ["Murder", "Serious crimes", "Rape"]})
    out = exclude_crime_categories(["Serious crimes"]).fit_transform(df)
    assert set(out["Type of Crime"]) == {"Murder", "Rape"}
    # empty exclusion list is a no-op
    out2 = exclude_crime_categories([]).fit_transform(df)
    assert len(out2) == 3
