import data_cleaning as dc
import feature_engineering as fe
import mlflow
import os.path as op
import pandas as pd
import scoring as sc
import training as tr

from ta_lib.core.api import register_processor


@register_processor("mlflow", "run-mlflow")
def run_mlflow(context, params):
    """Track and run data parameters and model metrics"""

    mlflow.set_tracking_uri(params["remote_server_uri"])
    mlflow.set_experiment(params["exp_name"])

    with mlflow.start_run(run_name="main") as main:
        with mlflow.start_run(run_name="data_cleaning", nested=True) as child_1:
            housing_df_clean = dc.clean_housing_table(context, {})
            mlflow.log_param(
                key="Clean Housing Table shape", value=housing_df_clean.shape
            )

            dc.create_training_datasets(
                context, {"test_size": 0.2, "target": "median_house_value"}
            )

        with mlflow.start_run(run_name="feature_engineering", nested=True) as child_2:
            fe.transform_features(
                context,
                {
                    "outliers": {"method": "mean", "drop": False},
                    "sampling_fraction": 0.1,
                },
            )

        with mlflow.start_run(run_name="training", nested=True) as child_3:
            tr.train_model(
                context,
                {
                    "sampling_fraction": 0.1,
                    "xgb": {
                        "gamma": 0.03,
                        "min_child_weight": 6,
                        "learning_rate": 0.1,
                        "max_depth": 5,
                        "n_estimators": 500,
                    },
                },
            )

        with mlflow.start_run(run_name="scoring", nested=True) as child_4:
            sc.score_model(context, {})
            target = pd.read_parquet("../data/test/housing/target.parquet")
            scores = pd.read_parquet("../data/test/housing/scored_output.parquet")

            score = (
                ((target["median_house_value"] - scores["yhat"]) ** 2).mean()
            ) ** 0.5
            mlflow.log_metric(key="model_rmse", value=score)
