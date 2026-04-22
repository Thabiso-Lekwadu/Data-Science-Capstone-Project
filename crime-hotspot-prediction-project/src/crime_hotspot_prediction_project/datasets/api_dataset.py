import pandas as pd
from kedro.io import AbstractDataset
from quantec.easydata.client import Client
import os

class APIDataset(AbstractDataset):
    def __init__(self, credentials: dict, selection_pk: int, freq: str = "A"):
        self._api_key = credentials["api_key"]
        self._url = credentials["url"]
        self._selection_pk = selection_pk
        self._freq = freq

    def _load(self) -> pd.DataFrame:
        client = Client(
            api_key=self._api_key,
            api_url=self._url,
            use_cache=True,
            cache_dir="cache",
        )
        df = client.get_data(selection_pk=self._selection_pk, freq=self._freq)

        # Ensure df is a DataFrame (some APIs return other formats)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        return df

    def _save(self, data: pd.DataFrame) -> None:
        raise NotImplementedError("Saving not supported for this dataset")

    def _describe(self) -> dict:
        return {
            "url": self._url,
            "selection_pk": self._selection_pk,
            "freq": self._freq,
        }