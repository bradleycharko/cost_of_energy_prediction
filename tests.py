# tests/test_data_ingestion.py

import pytest
from src.data_ingestion import fetch_energy_data

def test_fetch_energy_data():
    df = fetch_energy_data({"frequency": "hourly", "data_type": "price"})
    # Depending on your API, you may mock the response for consistent tests.
    assert df is not None
    assert isinstance(df.columns, pd.Index)
