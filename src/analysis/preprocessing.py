import sys
sys.path.append('../')

import os
import ta
import fire
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Union

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import Config
from src.base.logger import get_console_logger

logger = get_console_logger()


def transform_ts_data_into_features_and_target(
    path_to_input: Optional[Path] = os.path.join(Config.FILES["PREPROCESS_DATA"], "stocks", f"{Config.FILES['PREPROCESS_STOCK_FILE']}.parquet"),
    input_seq_len: Optional[int] = Config.ANALYSIS_CONFIG["INPUT_SEQ_LEN"],
    step_size: Optional[int] = Config.ANALYSIS_CONFIG["STEP_SIZE"]
    ) -> Tuple[pd.DataFrame, pd.Series]:
    """Pipeline in transforming time series data into features and target in pandas dataframe and numpy array format

    Args:
        path_to_input (Optional[Path], optional): Input path for data . Defaults to os.path.join(self.FILES["PREPROCESS_DATA"], "crypto", "Config.FILES['PREPROCESS_CRYPTO_FILE'].parquet").
        input_seq_len (Optional[int], optional): Input sequence length of the input features. Defaults to 24.
        step_size (Optional[int], optional): Steps size of the input features and target variable. Defaults to 1.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: _description_
    """
    ts_data = pd.read_parquet(path_to_input)
    ts_data = ts_data[Config.vars(["STOCKS"])]
    try:
        ts_data["time"] = ts_data["time"].apply(lambda x: datetime.fromtimestamp(x))
    except:
        logger.info("time column already in datetime format")
    ts_data.sort_values(by=["time"], inplace=True)
    
    features = pd.DataFrame()
    targets = pd.DataFrame()
    
    indices = get_cut_off_indices_features_and_target(ts_data, input_seq_len, step_size)
    
    n_examples = len(indices)
    x = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float32)
    y = np.ndarray(shape=(n_examples), dtype=np.float32)
    times = []
    for i, idx in enumerate(indices):
        x[i, :] = ts_data.iloc[idx[0]:idx[1]]["close"].values
        y[i] = ts_data.iloc[idx[1]:idx[2]]["close"].values
        times.append(ts_data.iloc[idx[1]]["time"])
        
    features = pd.DataFrame(
        x,
        columns=[f"price_{i+1}_hour_ago" for i in reversed(range(input_seq_len))]
    )
    
    targets = pd.DataFrame(y, columns=[f"target_price_next_hour"])
    
    return features, targets["target_price_next_hour"]


def get_cut_off_indices_features_and_target(data: pd.DataFrame,
                                            input_seq_len: int,
                                            step_size: int
                                            ) -> List[Tuple[int, int, int]]:
    """Create sequence features and target data 

    Args:
        data (pd.DataFrame): Input pandas dataframe
        input_seq_len (int): _description_
        step_size (int): _description_

    Returns:
        List[Tuple[int, int, int]]: _description_
    """
    logger.info("pre-compute cutoff indices to split dataframe rows at input sequence length and step size of {}".format(input_seq_len, step_size))
    stop_position = len(data) - 1
    
    subseq_first_idx = 0
    subseq_mid_idx = input_seq_len
    subseq_last_idx = input_seq_len + 1
    indices = []
    
    while subseq_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))
        subseq_first_idx += step_size
        subseq_mid_idx += step_size
        subseq_last_idx += step_size
        
    return indices


def get_price_columns(x: pd.DataFrame) -> List[str]:
    """Get the columns of the input dataframe that contain the price data

    Args:
        x (pd.DataFrame): Input features dataframe

    Returns:
        List[str]: List of columns that contain the substring 'price' data
    """
    return [col for col in x.columns if "price" in col]


class RSI(BaseEstimator, TransformerMixin):
    """
    Adds RSI to the input DataFrame from the `close` prices

    New columns are:
        - 'rsi'
    """
    def __init__(self, window: int = 14):
        self.window = window
        
        
    def fit(self,
            x: pd.DataFrame,
            y: Optional[Union[pd.DataFrame, pd.Series]] = None
            ) -> "RSI":
        """In this scenario, the fit method isn't doing anything. But it must be implemented. 
            This is a scenario of an estimator without parameters.
        """
        return self
    
    
    def _add_indicator(self, row: pd.Series) -> float:
        return pd.Series([ta.momentum.rsi(row, window=self.window)[-1]])
    
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute RSI and add it to the input dataframe

        Args:
            X (pd.DataFrame): Input features dataframe

        Returns:
            pd.DataFrame: Output input features dataframe with RSI added
        """
        logger.info("Adding RSI to the input DataFrame")
        df = X[get_price_columns(X)].apply(self._add_indicator, axis=1)
        df.columns = ["rsi"]
        X = pd.concat([X, df], axis=1)
        
        return X
    
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse the log of every cell of the DataFrame

        Args:
            X (pd.DataFrame): Input features dataframe

        Returns:
            pd.DataFrame: Output input features dataframe with RSI removed
        """
        X.drop(columns=['rsi'], inplace=True)
        
        return X
    
    
def get_price_percentage_return(X: pd.DataFrame, hours: int) -> pd.DataFrame:
    logger.info("Adding percentage return of price at hour {} to the input DataFrame".format(hours))
    X[f"percentage_return_{hours}_hour"] = (X["price_1_hour_ago"] - X[f"price_{hours}_hour_ago"])/ X[f"price_{hours}_hour_ago"]
    
    return X


def get_subset_of_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X[["price_1_hour_ago", "percentage_return_2_hour", "percentage_return_12_hour", "percentage_return_24_hour", "rsi"]]
    return X


def get_preprocessing_pipeline(pp_rsi_window: int=14) -> Pipeline:
    return make_pipeline(
        RSI(window=pp_rsi_window),
        FunctionTransformer(get_price_percentage_return, kw_args={"hours": 2}),
        FunctionTransformer(get_price_percentage_return, kw_args={"hours": 12}),
        FunctionTransformer(get_price_percentage_return, kw_args={"hours": 24}),
        FunctionTransformer(get_subset_of_features)
    )
    
    
if __name__ == "__main__":
    features, target = fire.Fire(transform_ts_data_into_features_and_target)
    
    preprocessing_pipeline = get_preprocessing_pipeline()
    
    # preprocessing_pipeline.fit(features)
    X = preprocessing_pipeline.transform(features)
    
    X.to_parquet(os.path.join(Config.FILES["PREPROCESS_DATA"], "stocks", "features.parquet"), index=False)
    pd.DataFrame({target.name: target}).to_parquet(os.path.join(Config.FILES["PREPROCESS_DATA"], "stocks", "target.parquet"), index=False)
    
    logger.info("done creating features and target data")