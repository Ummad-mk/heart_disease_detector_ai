"""
Unit tests for src.titanic.data
================================
Run:  pytest tests/titanic/test_data.py -v
"""
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import pytest

from src.titanic.data import load_raw, preprocess


class TestLoadRaw:
    def test_returns_dataframe(self):
        df = load_raw()
        assert isinstance(df, pd.DataFrame)

    def test_has_survived_column(self):
        df = load_raw()
        assert "survived" in df.columns

    def test_minimum_row_count(self):
        df = load_raw()
        assert len(df) >= 800, "Titanic dataset should have at least 800 rows"


class TestPreprocess:
    def test_split_sizes_correct(self):
        df = load_raw()
        X_train, X_test, y_train, y_test, encoders = preprocess(df)
        total = len(X_train) + len(X_test)
        assert abs(len(X_test) / total - 0.2) < 0.05

    def test_no_nulls_after_preprocess(self):
        df = load_raw()
        X_train, X_test, y_train, y_test, _ = preprocess(df)
        assert X_train.isnull().sum().sum() == 0
        assert X_test.isnull().sum().sum() == 0

    def test_encoders_returned(self):
        df = load_raw()
        _, _, _, _, encoders = preprocess(df)
        assert "sex" in encoders
        assert "embarked" in encoders
