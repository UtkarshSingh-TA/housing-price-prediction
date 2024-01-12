"""Processors for the model training step of the worklow."""
import logging
import os.path as op
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from ta_lib.core.api import (
    DEFAULT_ARTIFACTS_PATH,
    get_dataframe,
    get_feature_names_from_column_transformer,
    get_package_path,
    load_dataset,
    load_pipeline,
    register_processor,
    save_pipeline,
)
from ta_lib.regression.api import SKLStatsmodelOLS

logger = logging.getLogger(__name__)


@register_processor("model-gen", "train-model")
def train_model(context, params):
    """Train a regression model."""
    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    input_features_ds = "train/housing/features"
    input_target_ds = "train/housing/target"

    # load training datasets
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)
    num_columns = train_X.select_dtypes("number").columns

    # load pre-trained feature pipelines and other artifacts
    curated_columns = load_pipeline(op.join(artifacts_folder, "curated_columns.joblib"))
    features_transformer = load_pipeline(op.join(artifacts_folder, "features.joblib"))

    # sample data if needed. Useful for debugging/profiling purposes.
    sample_frac = params.get("sampling_fraction", None)
    if sample_frac is not None:
        logger.warn(f"The data has been sample by fraction: {sample_frac}")
        sample_X = train_X.sample(frac=sample_frac, random_state=context.random_seed)
    else:
        sample_X = train_X
    sample_y = train_y.loc[sample_X.index]

    # transform the training data
    train_X = get_dataframe(
        features_transformer.fit_transform(train_X, train_y),
        list(features_transformer.transformers_[0][1][0].get_feature_names())
        + list(num_columns),
    )
    train_X = train_X[curated_columns]

    # create training pipeline
    xgb_ppln = Pipeline(
        [
            (
                "estimator",
                XGBRegressor(
                    gamma=params["xgb"]["gamma"],
                    min_child_weight=params["xgb"]["min_child_weight"],
                    learning_rate=params["xgb"]["learning_rate"],
                    max_depth=params["xgb"]["max_depth"],
                    n_estimators=params["xgb"]["n_estimators"],
                ),
            )
        ]
    )

    # fit the training pipeline
    xgb_ppln.fit(train_X, train_y.values.ravel())

    # save fitted training pipeline
    save_pipeline(
        xgb_ppln, op.abspath(op.join(artifacts_folder, "train_pipeline.joblib"))
    )
