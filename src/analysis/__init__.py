import sys
sys.path.append(r"C:\Users\kewjs\Documents\02-Self_Learning\01-Data_Science\15-Stocks_Forecasting\src")

import os
import ta
import math
import holidays
import fnmatch
import numpy as np
import pandas as pd
from typing import Optional, Union
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.config import Config
from src.analysis.scraper import Scraper

class Logger(object):
    info     = print
    warning  = print
    error    = print
    critical = print 


class Analysis(Config):
    data = {}
    
    def __init__(self, city=["*"], logger=Logger()):
        self.city = city
        self.logger = logger
        self.scraper = Scraper(logger)
        self.us_holidays = holidays.US()
    
    
    def get_data(self) -> None:
        """Main function to get data from different sources
        """
        self.logger.info("Get KLSE stock data...")
        self.data["raw_klse_stock"] = self.scraper.get_klse_stock_data(
            company_list=self.SCRAPER_CONFIG["KLSE_COMPANY_LIST"],
            company_name=self.SCRAPER_CONFIG["KLSE_COMPANY_NAME"]
            )
        
        self.logger.info("Prepare data for modelling...")
        self.data["klse_stock"] = pd.DataFrame()
        for i, data in self.data["raw_klse_stock"].groupby(["name"]):
            data = data.set_index("date")
            data = self._generate_time_lags(df=data, col="close", lags=7)
            data = self._generate_cyclical_features(df=data, col="close", period=7, start_num=0)
            self.data["klse_stock"] = pd.concat([self.data["klse_stock"], data])
            
        file = os.path.join(self.FILES["PREPROCESS_DATA"], "stocks", "{}.parquet".format(self.FILES["PREPROCESS_STOCK_FILE"]))
        self.data["klse_stock"].to_parquet(file)
        
        self.logger.info("done...")
            
        return
    
    
    def _feature_engineer_for_univariate(self, df:pd.DataFrame) -> pd.DataFrame:
        df = self._generate_time_lags(df, self.ANALYSIS_CONFIG["LAGS_WINDOW"])
        
        df = (
            df
            .assign(hour = df.index.hour)
            .assign(day = df.index.day)
            .assign(month = df.index.month)
            .assign(day_of_week = df.index.dayofweek)
            )
        
        df = self._generate_cyclical_features(df, "hour", 24, 0)
        df = self._generate_cyclical_features(df, "month", 7, 0)
        df = self._generate_cyclical_features(df, "day_of_week", 12, 1)
        
        df = self._add_holiday_col(df, self.us_holidays)
            
        return df
    
    
    def _onehot_encode_pd(self, df:pd.DataFrame, cols:list[str]) -> pd.DataFrame:
        """One hot encode the input features

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        cols : list[str]
            List of columns to be one-hot encoded

        Returns
        -------
        pd.DataFrame
            Output dataframe with one-hot encoded columns
        """
        for col in cols:
            dummies = pd.get_dummies(df[col], prefix=col)
        
        return pd.concat([df, dummies], axis=1).drop(columns=cols)
    
    
    def _scaler(self, df:pd.DataFrame, cols:list[str], scaler_type:str="standard") -> pd.DataFrame:
        """Scale the input features

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        cols : list[str]
            List of columns to be scaled
        scaler_type : str, optional
            Type of scaler to be used, by default "standard"

        Returns
        -------
        pd.DataFrame
            Output dataframe with scaled columns
        """
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError("Invalid scaler type")
        
        df[cols] = scaler.fit_transform(df[cols])
        
        return df
    
    
    def _generate_time_lags(self, df: pd.DataFrame, col: str, lags: int) -> pd.DataFrame:
        """Create lags for the stock price data

        Args:
            df (pd.DataFrame): Input pandas dataframe
            col (str): Column name to create lag
            lags (int): Number of steps for the lag

        Returns:
            pd.DataFrame: Output dataframe with the time lags
        """
        for i in range(1, lags+1):
            df[f"{col}_lag_t{i}"] = df[col].shift(i)
            
        return df
    

    def _generate_cyclical_features(self, df:pd.DataFrame, col:str, period:int, start_num:int=0):
        """Create cyclical features from hour column

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with hour column
        col : str
            'hour' column
        period : int
            Period of 'hour' to be considered to create cyclical features 
        start_num : int, optional
            Start number to create cyclical feature, by default 0

        Returns
        -------
        pd.DataFrame
            Dataframe with sin and cos of 'col' column
        """
        kwargs = {
            f"sin_{col}" : lambda x: np.sin(2*np.pi*(df[col]-start_num)/period),
            f"cos_{col}" : lambda x: np.cos(2*np.pi*(df[col]-start_num)/period)    
                }
        return df.assign(**kwargs)
    
    
    def _is_holiday(self, date):
        """Create holiday flagging column

        Parameters
        ----------
        date : _type_
            _description_

        Returns
        -------
        int
            1 if it is holiday 0 if not holiday
        """
        date = date.replace(hour = 0)
        return 1 if (date in self.us_holidays) else 0


    def _add_holiday_col(self, df:pd.DataFrame, holidays):
        """Add holiday flagging  column into dataframe

        Parameters
        ----------
        df : pd.DataFrame
            _description_
        holidays : _type_
            Holidays object from holiday package

        Returns
        -------
        pd.DataFrame
            Dataframe with holiday column called is_holiday
        """
        return df.assign(is_holiday=df.index.to_series().apply(self._is_holiday))
    
    
    def adf_test(self, df:pd.DataFrame, signif:float=0.05):
        """Generate Augmented Dickey Fuller Test (ADF)

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe contains only numeric columns for ADF test
        signif : float, optional
            Significant values to test ADF stationarity, by default 0.05
        """
        dftest = adfuller(df, autolag='AIC')
        adf = pd.Series(dftest[0:4], index=["Test Statistic", "p-value", "#Lags Used", "# Observations"])
        for key, value in dftest[4].items():
            adf['Critical Value (%s) '%key] = value
        print(adf)
        
        p = adf['p-value']
        if p <= signif:
            print(f"Series is Stationary")
        else:
            print(f"Series is Non-Stationary")
            
    
    def _ts_decomposition(self, x):
        sd_24 = seasonal_decompose(x, period=24)
        sd_168 = seasonal_decompose(x - sd_24.seasonal, period=168)
        sd_8766 = seasonal_decompose(x - sd_168.seasonal, period=math.floor(x.shape[0]/2))
        
        return sd_24, sd_168, sd_8766    
    
    
class RSI(Config, BaseEstimator, TransformerMixin):
    """
    Adds RSI to the input DataFrame from the `close` prices

    New columns are:
        - 'rsi'
    """
    def __init__(self, window: int=14, logger=Logger()):
        self.window = window
        self.logger = logger
        
        
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
        self.logger.info("Adding RSI to the input DataFrame")
        df = X[self.MODELLING_CONFIG["TARGET_VAR"]].apply(self._add_indicator, axis=1)
        df.columns = ["rsi"]
        X = pd.concat([X, df], axis=1)
        
        return X