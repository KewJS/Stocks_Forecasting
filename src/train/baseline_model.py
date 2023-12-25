import os
import pandas as pd
from comet_ml import Experiment
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Lasso
from typing import Dict, Union, Optional, Callable

from src.config import Config
from src.base.logger import get_console_logger
from src.analysis.preprocessing import transform_ts_data_into_features_and_target

logger = get_console_logger()


def get_baseline_model_error(X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Get baseline model error

    Args:
        X_test (pd.DataFrame): X_test dataframe
        y_test (pd.Series): y_test series

    Returns:
        float: _description_
    """
    predictions = X_test["price_1_hour_ago"]
    mae = mean_absolute_error(y_test, predictions)
    
    return mae


def train(X: pd.DataFrame, y: pd.Series) -> None:
    """Train a boosting model using the input features 'X' and target 'y'

    Args:
        X (pd.DataFrame): Input dataframe
        y (pd.Series): Target feature
    """
    logger.info("building baseline model by comparing the price with 1 hour lag")
    experiment = Experiment(
        api_key = os.environ["COMET_ML_API_KEY"],
        workspace = os.eviron["COMET_ML_WORKSPACE"],
        project_name = "prediction-of-cryptocurrency-prices",
    )
    experiment.add_tag("baseline_model")
    
    logger.ingo("splitting data into train and test sets at the ratio {}".format(Config.MODELLING_CONFIG["SPLIT_RATIO"]))
    train_sample_data = int(Config.MODELLING_CONFIG["SPLIT_RATIO"] * len(X))
    X_train, X_test = X[:train_sample_data], X[train_sample_data:]
    y_train, y_test = y[:train_sample_data], y[train_sample_data:]
    logger.info(f"Train sample size: {len(X_train)}")
    logger.info(f"Test sample size: {len(X_test)}")
    
    baseline_train_mae = get_baseline_model_error(X_train, y_train)
    logger.info(f"Train MAE: {baseline_train_mae}")
    experiment.log_metrics({"Train MAE": baseline_train_mae})
    
    baseline_test_mae = get_baseline_model_error(X_test, y_test)
    logger.info(f"Test MAE: {baseline_test_mae}")
    experiment.log_metrics({"Test MAE": baseline_test_mae})
    
    
if __name__ == "__main__":
    logger.info("Generating features and targets")
    features, target = transform_ts_data_into_features_and_target()
    
    logger.info("Starting training")
    train(features, target)