"""
Unit tests for src.heart_disease.data
======================================
Run:  pytest tests/heart_disease/test_data.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import pytest

from src.heart_disease.data import load_raw, preprocess
from src.heart_disease import config


class TestLoadRaw:
    def test_returns_dataframe(self):
        df = load_raw()
        assert isinstance(df, pd.DataFrame)

    def test_has_target_column(self):
        df = load_raw()
        assert "target" in df.columns

    def test_has_required_features(self):
        df = load_raw()
        for feature in config.FEATURES:
            assert feature in df.columns, f"Feature '{feature}' missing from dataset"

    def test_minimum_row_count(self):
        df = load_raw()
        assert len(df) >= 300, "Heart Disease dataset should have at least 300 rows"


class TestPreprocess:
    def test_split_sizes_correct(self):
        df = load_raw()
        X_train, X_test, y_train, y_test, _ = preprocess(df)
        total = len(X_train) + len(X_test)
        assert abs(len(X_test) / total - 0.2) < 0.05

    def test_no_nulls_after_preprocess(self):
        df = load_raw()
        X_train, X_test, _, _, _ = preprocess(df)
        assert X_train.isnull().sum().sum() == 0
        assert X_test.isnull().sum().sum() == 0

    def test_only_selected_features_in_X(self):
        df = load_raw()
        X_train, _, _, _, _ = preprocess(df)
        assert list(X_train.columns) == config.FEATURES

    def test_cp_encoder_returned(self):
        df = load_raw()
        _, _, _, _, encoders = preprocess(df)
        assert "cp" in encoders
