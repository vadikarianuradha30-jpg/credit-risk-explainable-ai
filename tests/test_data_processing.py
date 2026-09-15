import sys
import os

# Allow imports from the src folder
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data_processing import load_and_preprocess


def test_train_test_split_sizes():
    """Train and test sets should add up to the total number of rows, and match demo_test length."""
    X_train, X_test, y_train, y_test, demo_test = load_and_preprocess()

    total_rows = len(X_train) + len(X_test)
    assert total_rows == 1000, f"Expected 1000 total rows, got {total_rows}"
    assert len(X_test) == len(demo_test), "X_test and demo_test should have the same number of rows"
    assert len(X_train) == len(y_train), "X_train and y_train should have the same number of rows"
    assert len(X_test) == len(y_test), "X_test and y_test should have the same number of rows"


def test_no_missing_values_after_preprocessing():
    """After preprocessing, there should be no missing (NaN) values in the feature set."""
    X_train, X_test, y_train, y_test, demo_test = load_and_preprocess()

    assert X_train.isnull().sum().sum() == 0, "X_train should not contain missing values"
    assert X_test.isnull().sum().sum() == 0, "X_test should not contain missing values"


def test_demographic_columns_separated():
    """Age and Sex should NOT be used as prediction features, but should exist in demo_test for the fairness audit."""
    X_train, X_test, y_train, y_test, demo_test = load_and_preprocess()

    assert "Age" in demo_test.columns, "Age should be present in demo_test"
    assert "Sex" in demo_test.columns, "Sex should be present in demo_test"


def test_target_values_are_binary():
    """The target column (Risk) should only contain 0s and 1s after encoding."""
    X_train, X_test, y_train, y_test, demo_test = load_and_preprocess()

    unique_train_values = set(y_train.unique())
    unique_test_values = set(y_test.unique())

    assert unique_train_values.issubset({0, 1}), f"y_train has unexpected values: {unique_train_values}"
    assert unique_test_values.issubset({0, 1}), f"y_test has unexpected values: {unique_test_values}"