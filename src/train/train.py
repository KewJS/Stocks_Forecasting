import sys
sys.path.append(r"../")

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, Union, Optional, Callable

from comet_ml import Experiment
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.config import Config
from src.analysis.preprocessing import (
    transform_ts_data_into_features_and_target,
    get_preprocessing_pipeline
)
from src.train.hyperparams import find_best_hyperparams
from src.base.logger import get_console_logger

logger = get_console_logger()


def get_baseline_model_error(X_test: pd.DataFrame, 
                             y_test: pd.Series) -> float:
    """Return the baseline model errors"""
    predictions = X_test["price_1_hour_ago"]
    
    return mean_absolute_error(y_test, predictions)


def get_model_fn_from_name(model_name: str) -> Callable:
    """Returns the model function given the model name"""
    if model_name == "lasso":
        return Lasso
    elif model_name == "xgboost":
        return XGBRegressor
    elif model_name == "lightgbm":
        return LGBMRegressor
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    
def train(x: pd.DataFrame,
          y: pd.Series,
          model: str,
          tune_hyperparam: Optional[bool] = False,
          hyperparam_trials: Optional[int] = 10,
          ) -> None:
    """ Train a boosting tree model using the input features `X` and targets `y`,
    possibly running hyperparameter tuning.

    Args:
        x (pd.DataFrame): Input X features in pandas dataframe
        y (pd.Series): Input y target in pandas series
        model (str): Model name
        tune_hyperparam (Optional[bool], optional): Boolean option to either run hyperparameter tuning or not. Defaults to False.
        hyperparam_trials (Optional[int], optional): Trials to run when tuning the hyperparameters. Defaults to 10.
    """
    model_fn = get_model_fn_from_name(model)
    
    experiment = Experiment(
        api_key=os.environ["COMET_ML_API_KEY"],
        workspace=os.environ["COMET_ML_WORKSPACE"],
        project_name=os.environ["COMET_ML_PROJECT_NAME"],
    )
    experiment.add_tag(model)
    
    train_sample_size = int(Config.MODELLING_CONFIG["SPLIT_RATIO"] * len(x))
    X_train, X_test = x[:train_sample_size], x[train_sample_size:]
    y_train, y_test = y[:train_sample_size], y[train_sample_size:]
    logger.info("Train sample size: {}".format(len(X_train)))
    logger.info("Test sample size: {}".format(len(X_test)))
    
    if not tune_hyperparam:
        logger.info("Using default hyperparameters")
        pipeline = make_pipeline(
            get_preprocessing_pipeline(),
            model_fn()
        )
    else:
        logger.info("Finding the best hyperparameters with cross-validation")
        best_preprocessing_hyperparams, best_model_hyperparams = find_best_hyperparams(
            model_fn, hyperparam_trials, X_train, y_train, experiment
        )
        logger.info("Best preprocessing hyperparameters: {}".format(best_preprocessing_hyperparams))
        logger.info("Best model hyperparameters: {}".format(best_model_hyperparams))
        
        pipeline = make_pipeline(
            get_preprocessing_pipeline(**best_preprocessing_hyperparams),
            model_fn(**best_model_hyperparams)
        )
        
        experiment.add_tag("hyperparameters-tuning")
    
    logger.info("Fitting model with default hyperparameters")
    pipeline.fit(X_train, y_train)
    
    train_pred = pipeline.predict(X_train)
    train_error = mean_absolute_error(y_train, train_pred)
    logger.info("Train MAE: {}".format(train_error))
    experiment.log_metrics({"Train MAE": train_error})
    
    test_pred = pipeline.predict(X_test)
    test_error = mean_absolute_error(y_test, test_pred)
    logger.info("Test MAE: {}".format(test_error))
    experiment.log_metrics({"Test MAE": test_error})
    
    if not Path(os.path.join(Config.FILES["MODELS_DIR"], "stocks")).exists():
        logger.info("Create models directory for 'stocks'")
        Path(os.path.join(Config.FILES["MODELS_DIR"], "stocks")).mkdir(parents=True)
    
    logger.info("Saving model to disk")
    with open(os.path.join(Config.FILES["MODELS_DIR"], "stocks", "{}_model.pkl".format(model)), "wb") as f:
        pickle.dump(pipeline, f)
        
    experiment.log_model(str(model_fn), str(os.path.join(Config.FILES["MODELS_DIR"], "stocks", "{}_model.pkl".format(model))))
    
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="lasso", help="Model name")
    parser.add_argument("--tune-hyperparam", action="store_true", help="Whether to tune hyperparameters")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--hyperparam-trials", type=int, default=10)
    args = parser.parse_args()
    
    logger.info("Generating features and targets")
    features, target = transform_ts_data_into_features_and_target()
    
    if args.sample_size is not None:
        features = features.head(args.sample_size)
        target = target.head(args.sample_size)
        
    logger.info("Training model")
    train(features,
          target,
          model=args.model,
          tune_hyperparam=args.tune_hyperparam,
          hyperparam_trials=args.hyperparam_trials
          )