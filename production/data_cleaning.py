"""Processors for the data cleaning step of the worklow.

The processors in this step, apply the various cleaning steps identified
during EDA to create the training datasets.
"""
import numpy as np
import pandas as pd
from scripts import binned_median_income
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit

from ta_lib.core.api import (
    custom_train_test_split,
    load_dataset,
    register_processor,
    save_dataset,
    string_cleaning,
)


@register_processor("data-cleaning", "housing")
def clean_housing_table(context, params):
    """Clean the ``HOUSING`` data table.

    The table contains information on the inventory being sold. This
    includes information on inventory id, properties of the item and
    so on.
    """

    input_dataset = "raw/housing"
    output_dataset = "cleaned/housing"

    # load dataset
    housing_df = load_dataset(context, input_dataset)

    housing_df_clean = (
        housing_df
        # set dtypes : nothing to do here
        .passthrough()
        .transform_columns(["ocean_proximity"], string_cleaning, elementwise=False)
        .replace({"": np.NaN})
        # clean column names (comment out this line while cleaning data above)
        .clean_names(case_type="snake")
    )

    # save the dataset
    save_dataset(context, housing_df_clean, output_dataset)

    return housing_df_clean


@register_processor("data-cleaning", "train-test")
def create_training_datasets(context, params):
    """Split the ``HOUSING`` table into ``train`` and ``test`` datasets."""

    input_dataset = "cleaned/housing"
    output_train_features = "train/housing/features"
    output_train_target = "train/housing/target"
    output_test_features = "test/housing/features"
    output_test_target = "test/housing/target"

    # load dataset
    housing_df_processed = load_dataset(context, input_dataset)

    # creating additional features that are not affected by train test split. These are features that are processed globally
    # first time customer(02_data_processing.ipynb)###############

    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room=True):
            self.add_bedrooms_per_room = add_bedrooms_per_room

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
            population_per_household = X[:, population_ix] / X[:, households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[
                    X, rooms_per_household, population_per_household, bedrooms_per_room
                ]

            else:
                return np.c_[X, rooms_per_household, population_per_household]

    col_add = CombinedAttributesAdder()
    housing_ext = col_add.fit_transform(housing_df_processed.values)

    housing_clean = pd.DataFrame(
        housing_ext,
        columns=list(housing_df_processed.columns)
        + ["rooms_per_household", "population_per_household", "bedrooms_per_room"],
    )

    # split the data
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=params["test_size"], random_state=context.random_seed
    )
    housing_df_train, housing_df_test = custom_train_test_split(
        housing_clean, splitter, by=binned_median_income
    )

    # split train dataset into features and target
    target_col = params["target"]
    train_X, train_y = (
        housing_df_train
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )

    # save the train dataset
    save_dataset(context, train_X, output_train_features)
    save_dataset(context, train_y, output_train_target)

    # split test dataset into features and target
    test_X, test_y = (
        housing_df_test
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )

    # save the datasets
    save_dataset(context, test_X, output_test_features)
    save_dataset(context, test_y, output_test_target)
