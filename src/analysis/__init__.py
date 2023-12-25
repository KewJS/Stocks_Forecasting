import os
import math
import glob
import holidays
import fnmatch
import numpy as np
import pandas as pd
from ftplib import FTP
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Any
from pvlib.iotools import read_surfrad

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot

from src.config import Config

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
        
        self.us_holidays = holidays.US()
        
    
    @staticmethod
    def vars(types=[], wc_vars=[], qreturn_dict=False):
        """ Return list of variable names
        
        Acquire the right features from dataframe to be input into model.  
        Features will be acquired based the value 'predictive' in the VARS dictionary. 

        Parameters
        ----------
        types : str
            VARS name on type of features
        
        Returns
        -------
        Features with predictive == True in self.VARS
        """
        if types==None:
            types = [V for V in Config.VARS]
        selected_vars = []
        for t in types:
            for d in Config.VARS[t]:
                if not d.get("predictive"):
                    continue
                if len(wc_vars) != 0: 
                    matched_vars = fnmatch.filter(wc_vars, d["var"])
                    if qreturn_dict:
                        for v in matched_vars:
                            dd = d.copy()
                            dd["var"] = v 
                            if not dd in selected_vars:
                                selected_vars.append(dd)
                    else:
                        for v in matched_vars:
                            if not v in selected_vars:
                                selected_vars.append(v)
                else:
                    if qreturn_dict and not d in selected_vars:
                        selected_vars.append(d)
                    else:
                        if not d["var"] in selected_vars:
                            selected_vars.append(d["var"])
        return selected_vars
    
    
    def _generate_merge_data(self) -> None:
        """Merging of all downloaded raw solar radiation data
        """
        for year in self.ANALYSIS_CONFIG["YEAR_LIST"]:
            merge_list = []
            for location in self.ANALYSIS_CONFIG["LOCATION_LIST"]:
                self.logger.info(f"  starting dataframe collection from {location} at {year}...")
                path = os.path.join(self.FILES["RAW_DATA"], location, year)
                filenames = glob.glob(path + "/*.dat")
                temp_df = self._merge_data(filenames)
                temp_df["city"] = location
                merge_list.append(temp_df)

            self.data[f"merge_{year}"] = pd.concat(merge_list, axis=0, ignore_index=True)
        
            if self.QDEBUG:
                self.logger.info("  export merged city dataframe from year: {}".format(year))
                self.data[f"merge_{year}"].to_parquet(os.path.join(self.FILES["PREPROCESS_DATA"], f"merge_{year}.parquet"))
                
        return
    
    
    def _merge_data(self, filenames:str) -> pd.DataFrame:
        """Simple function to merge all the dataframes by year
        'read_surfrad' function gives two outputs, a dataframe and a metadata.

        Parameters
        ----------
        filenames : str
            File name of the scraped NOAA sun radiation data

        Returns
        -------
        pd.DataFrame
            Output dataframe with merged from raw data
        """
        df_list = []   
        for filename in filenames:
            df, _ = read_surfrad(filename, map_variables=True)
            df_list.append(df)        
        frame = pd.concat(df_list, axis=0, ignore_index=True)
        
        return frame
    
    
    def get_data(self) -> None:
        if self.ANALYSIS_CONFIG["READ_MERGE"]:
            self.logger.info("  read in merged solar radiation file...")
            merge_files_list = [file for file in os.listdir(self.FILES["MERGE_PREPROCESS_DATA"]) if ".parquet" in file]
            for file in merge_files_list:
                self.data[file.split(".")[0]] = pd.read_parquet(os.path.join(self.FILES["MERGE_PREPROCESS_DATA"], file))
        else:
            self.logger.info("  read in raw solar radiation file within year of {}...".format(self.ANALYSIS_CONFIG["YEAR_LIST"]))
            self._generate_merge_data()
            
        self.logger.info("  updating dataframe datetime index to UTC localized format and drop individual time components...")
        for k in self.data.keys():
            self.logger.info("    > {}".format(k))
            self.data[k] = self._format_index(self.data[k])
            self.data[k].drop(columns=["year", "month", "day", "minute", "minute"], inplace=True)
            self.data[k] = self.data[k][self.vars(["RADIATION"], self.data[k])]
            
        self.logger.info("  create train & test dataset at ratio {}...".format(self.MODELLING_CONFIG["SPLIT_RATIO"]))
        train_list = list(self.data.keys())[:int(len(self.data.keys()) - len(self.data.keys())*0.2)]
        val_list = [list(self.data.keys())[int(len(self.data.keys()) - len(self.data.keys())*0.2)]]
        test_list = list(self.data.keys())[-int(len(self.data.keys())*0.2):]

        self.logger.info("    > train data from {}".format(train_list))
        self.merge_train = pd.DataFrame()
        for train in train_list:
            self.merge_train = pd.concat([self.merge_train, self.data[train]])
            
        self.logger.info("    > validation data from {}".format(val_list))
        self.merge_val = pd.DataFrame()
        for val in val_list:
            self.merge_val = pd.concat([self.merge_val, self.data[val]])
        
        self.logger.info("    > test data from {}".format(test_list))
        self.merge_test = pd.DataFrame()
        for test in test_list:
            self.merge_test = pd.concat([self.merge_test, self.data[test]])

        self.merge_train, self.merge_val, self.merge_test = self.merge_train_val_test(merge_train=self.merge_train,
                                                                                      merge_val=self.merge_val,
                                                                                      merge_test=self.merge_test,
                                                                                      sample_rate=self.ANALYSIS_CONFIG["SAMPLE_RATE"])
        
        self.merge_train, self.merge_val, self.merge_test = self.get_univariate_target(train_df=self.merge_train,
                                                                                       val_df=self.merge_val,
                                                                                       test_df=self.merge_test)
                 
        if self.QDEBUG:
            self.logger.info("  export 'merge' train, validation and test data at interval {}...".format(self.ANALYSIS_CONFIG["SAMPLE_RATE"]))
            self.merge_train.to_parquet(os.path.join(self.FILES["PREPROCESS_DATA"], "merge_train_{}.parquet".format(self.ANALYSIS_CONFIG["SAMPLE_RATE"])))
            self.merge_val.to_parquet(os.path.join(self.FILES["PREPROCESS_DATA"], "merge_val_{}.parquet".format(self.ANALYSIS_CONFIG["SAMPLE_RATE"])))
            self.merge_test.to_parquet(os.path.join(self.FILES["PREPROCESS_DATA"], "merge_test_{}.parquet".format(self.ANALYSIS_CONFIG["SAMPLE_RATE"])))
            
        return
    
    
    def merge_train_val_test(self, merge_train:pd.DataFrame, merge_val:pd.DataFrame, merge_test:pd.DataFrame, sample_rate:str="daily"):
        """Create train, validation and test data for 'daily' and 'hourly' sample dataset

        Parameters
        ----------
        merge_train : pd.DataFrame
            Merged train dataset
        merge_val : pd.DataFrame
            Merged validation dataset
        merge_test : pd.DataFrame
            Merged test dataset

        Returns
        -------
        pd.DataFrame
            Train, validation and test dataset for 'daily' and 'hourly'
        """
        train_dict, val_dict, test_dict = {}, {}, {}
        if sample_rate == "daily":
            for city in merge_train["city"].unique():
                train_dict[city], val_dict[city], test_dict[city] = self._resample_data(train_df=merge_train,
                                                                                        val_df=merge_val,
                                                                                        test_df=merge_test, 
                                                                                        location=city, frequency="daily")
                train_dict[city]["city"], val_dict[city]["city"], test_dict[city]["city"] = city, city, city
        
        elif sample_rate == "hourly":
            for city in merge_train["city"].unique():
                train_dict[city], val_dict[city], test_dict[city] = self._resample_data(train_df=merge_train, 
                                                                                        val_df=merge_val,
                                                                                        test_df=merge_test, 
                                                                                        location=city, frequency="hourly")
                train_dict[city]["city"], val_dict[city]["city"], test_dict[city]["city"] = city, city, city
        
        self.logger.info("  create train, validation and test data at '{}' interval...".format(sample_rate))
        merge_train = pd.concat(train_dict.values())
        merge_val = pd.concat(val_dict.values())
        merge_test = pd.concat(test_dict.values())
        
        self.logger.info("  missing values interpolations...")
        for city in merge_train["city"].unique():
            merge_train[merge_train["city"]==city] = merge_train[merge_train["city"]==city].interpolate(method="linear")
        
        return merge_train, merge_val, merge_test
    
    
    def _resample_data(self, train_df:pd.DataFrame, val_df:pd.DataFrame, test_df:pd.DataFrame, 
                       location:str="Bondville_IL", frequency:str="daily"):
        """Simple function to split data into training/test/validation set
        Also, split based on hourly/daily/monthly aggregates

        Parameters
        ----------
        train_df : pd.DataFrame
            Train dataset
        val_df : pd.DataFrame
            Validation dataset
        test_df : pd.DataFrame
            Test dataset
        location : str, optional
            City location, by default "Bondville_IL"
        frequency : str, optional
            Data sample frequency, from 'hourly', 'daily' and 'monthly', defaults to "daily", by default "daily"

        Returns
        -------
        pd.DataFrame
            Resampled train, validation and test dataset
        """        
        if frequency == "hourly":
            train = train_df[train_df["city"]==location][train_df.columns.difference(["city"])].resample("H").mean()       
            val = val_df[val_df["city"]==location][val_df.columns.difference(["city"])].resample("H").mean()
            test = test_df[test_df["city"]==location][test_df.columns.difference(["city"])].resample("H").mean()
            
        elif frequency == "daily":
            train = train_df[train_df["city"]==location][train_df.columns.difference(["city"])].resample("D").mean()
            val = val_df[val_df["city"]==location][val_df.columns.difference(["city"])].resample("D").mean()
            test = test_df[test_df["city"]==location][test_df.columns.difference(["city"])].resample("D").mean()
        
        elif frequency == "monthly":
            train = train_df[train_df["city"]==location][train_df.columns.difference(["city"])].resample("MS").mean()
            val = val_df[val_df["city"]==location][val_df.columns.difference(["city"])].resample("MS").mean()
            test = test_df[test_df["city"]==location][test_df.columns.difference(["city"])].resample("MS").mean()  
            
        return train, val, test
    
    
    def get_univariate_target(self, train_df:pd.DataFrame, val_df:pd.DataFrame, test_df:pd.DataFrame):
        self.logger.info("  get the univariate target feature '{}'...".format(self.ANALYSIS_CONFIG["TARGET_VAR"]))
        
        train_uni, val_uni, test_uni = {}, {}, {}
        for city in train_df["city"].unique(): 
            train_uni[city] = self._feature_engineer_for_univariate(df=train_df[train_df["city"]==city])
            val_uni[city] = self._feature_engineer_for_univariate(df=val_df[val_df["city"]==city])
            test_uni[city] = self._feature_engineer_for_univariate(df=test_df[test_df["city"]==city])
        
        merge_train = pd.concat(train_uni.values())
        merge_val = pd.concat(val_uni.values())
        merge_test = pd.concat(test_uni.values())
            
        return merge_train, merge_val, merge_test
    
    
    def _feature_engineer_for_univariate(self, df:pd.DataFrame) -> pd.DataFrame:
        df = self._generate_time_lags(df, self.MODELLING_CONFIG["LAGS_WINDOW"])
        
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
    
    
    def _generate_time_lags(self, df:pd.DataFrame, n_lags:int) -> pd.DataFrame:
        """Generate time lags features for 'netsolar'

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with 'netsolar' column
        n_lags : int
            Lags intended

        Returns
        -------
        pd.DataFrame
            Dataframe with lags of 'netsolar'
        """
        df_n = df.copy()
        for n in range(1, n_lags+1):
            df_n["{}_lags{}".format(self.ANALYSIS_CONFIG["TARGET_VAR"], n)] = df_n[self.ANALYSIS_CONFIG["TARGET_VAR"]].shift(n)
        
        return df_n
    
    
    def _onehot_encode_pd(self, df:pd.DataFrame, cols:List[str]) -> pd.DataFrame:
        """One hot encode the input features

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        cols : List[str]
            List of columns to be one-hot encoded

        Returns
        -------
        pd.DataFrame
            Output dataframe with one-hot encoded columns
        """
        for col in cols:
            dummies = pd.get_dummies(df[col], prefix=col)
        
        return pd.concat([df, dummies], axis=1).drop(columns=cols)
    

    def _generate_cyclical_features(self, df:pd.DataFrame, col_name:str, period:int, start_num:int=0):
        """Create cyclical features from hour column

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with hour column
        col_name : str
            'hour' column
        period : int
            Period of 'hour' to be considered to create cyclical features 
        start_num : int, optional
            Start number to create cyclical feature, by default 0

        Returns
        -------
        pd.DataFrame
            Dataframe with sin and cos of 'hour' column
        """
        kwargs = {
            f"sin_{col_name}" : lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
            f"cos_{col_name}" : lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period)    
                }
        return df.assign(**kwargs).drop(columns=[col_name])
    
    
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
    
    
    def _format_index(self, data:pd.DataFrame) -> pd.DataFrame:
        """Create UTC localized DatetimeIndex for the dataframe.
        
        Parameters
        ----------
        data: Dataframe
            Must contain columns 'year', 'jday', 'hour' and
            'minute'.
        Return
        ------
        data: Dataframe
            Dataframe with a DatetimeIndex localized to UTC.
        """
        year = data["year"].apply(str)
        jday = data["jday"].apply(lambda x: "{:03d}".format(x))
        hours = data["hour"].apply(lambda x: "{:02d}".format(x))
        minutes = data["minute"].apply(lambda x: "{:02d}".format(x))
        index = pd.to_datetime(year + jday + hours + minutes, format="%Y%j%H%M")
        data.index = index
        data = data.tz_localize("UTC")
        
        return data
    
    
    def _netsolar_difference(self, df:pd.DataFrame) -> pd.DataFrame:
        """Create 'netsolar' differencing column at defined differencing order

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe that has the 'netsolar' column

        Returns
        -------
        pd.DataFrame
            Dataframe with differenced 'netsolar' column
        """
        netsolar_diff = np.diff(df["netsolar"], n=self.ANALYSIS_CONFIG["DIFFERENCE_ORDER"])
        if len(netsolar_diff) < len(df):
            netsolar_diff = np.append(netsolar_diff, [np.nan] * (len(df) - len(netsolar_diff)))
        
        df["netsolar_diff"] = netsolar_diff
        
        return df
    
    
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
    
    
    def _get_statistics(self, train:pd.DataFrame, train_h:pd.DataFrame):
        self.roll_mean_h, self.roll_std_h = {}, {}
        self.roll_mean_d, self.roll_std_d = {}, {}
        
        self.logger.info("  create 10 days rolling average and 24 hour rolling average...")
        for city in train_d["city"].unique():
            # 10 days rolling average
            self.roll_mean_d[city] = train_d[train_d["city"]==city][train_d.columns.difference(["city"])].rolling(window=10, center=False).mean()
            self.roll_std_d[city] = train_d[train_d["city"]==city][train_d.columns.difference(["city"])].rolling(window=10, center=False).std()
            self.roll_mean_d[city]["jday"] = train_d[train_d["city"]==city][train_d.columns.difference(["city"])]["jday"]
            self.roll_std_d[city]["jday"] = train_d[train_d["city"]==city][train_d.columns.difference(["city"])]["jday"]
            
            # 24 hour rolling average
            self.roll_mean_h[city] = train_h[train_h["city"]==city][train_h.columns.difference(["city"])].rolling(window=24, center=False).mean()
            self.roll_std_h[city] = train_h[train_h["city"]==city][train_h.columns.difference(["city"])].rolling(window=24, center=False).std()
            self.roll_mean_h[city]["jday"] = train_h[train_h["city"]==city][train_h.columns.difference(["city"])]["jday"]
            self.roll_std_h[city]["jday"] = train_h[train_h["city"]==city][train_h.columns.difference(["city"])]["jday"]


    def ts_components_plot(self, daily_comp, weekly_comp, yearly_comp, city=None):
        f, axes = plt.subplots(6, 1, figsize=(15,16))

        # # setting figure title and adjusting title position and size
        plt.suptitle("Summary of Seasonal Decomposition of {}".format(city), y=0.92, fontsize=15, fontweight="bold")
        plt.subplots_adjust(wspace=0.5, hspace=0.4)
        
        # # plotting trend component
        axes[0].plot(daily_comp.observed)
        axes[0].set_title("Observed component", fontdict={"fontsize": 12})

        # # plotting trend component
        axes[1].plot(daily_comp.trend)
        axes[1].set_title("Trend component", fontdict={"fontsize": 12})

        # # plotting daily seasonal component
        axes[2].plot(daily_comp.seasonal)
        axes[2].set_title("Daily seasonal component", fontdict={"fontsize": 12})

        # # plotting weekly seasonal component
        axes[3].plot(weekly_comp.seasonal)
        axes[3].set_title("Weekly seasonal component", fontdict={"fontsize": 12})

        # # plotting yearly seasonality
        axes[4].plot(yearly_comp.seasonal)
        axes[4].set_title("Yearly seasonal component", fontdict={"fontsize": 12})

        # # plotting residual of decomposition
        axes[5].plot(daily_comp.resid)
        axes[5].set_title("Residual component", fontdict={"fontsize": 12})

        for a in axes:
            a.set_ylabel("MW")
            
        return plt.show()
    
    
    # # EDA
    def lineplot_plotly(self, df, groupby_col, col):
        fig = go.Figure()

        # Iterate over unique groups
        for group in df[groupby_col].unique():
            group_df = df[df[groupby_col] == group]
            fig.add_trace(go.Scatter(
                x=group_df.index,
                y=group_df[col],
                name=f"{group}",
                mode="lines"
            ))

        # Add dropdown menu
        buttons = []
        for group in df[groupby_col].unique():
            visible = [True if g == group else False for g in df[groupby_col].unique()]
            buttons.append(dict(
                label=f"{group}",
                method="update",
                args=[{"visible": visible}, {"title": f"Group {group}"}]
            ))

        fig.update_layout(
            updatemenus=[go.layout.Updatemenu(
                buttons=buttons,
                active=0,
                direction="down",
                x=0.2,
                y=1.2,
            )],
            width=1000,
            height=400
        )

        fig.update_layout(
            title="<b>Line Plot of '{}'</b>".format(col),
            title_x=0.5, title_y=0.9,
            xaxis_title="datetime",
            yaxis_title="value"
        )

        # Show the plot
        return fig.show()
    
    
    def single_lineplot(self, df, title):
        data = []
        
        value = go.Scatter(
            x=df.index,
            y=df.netsolar,
            mode="lines",
            name="netsolar",
            marker=dict(),
            text=df.index,
            line=dict(color="rgba(0,0,0, 0.3)"),
        )
        data.append(value)

        layout = dict(
            title=title,
            xaxis=dict(title="Date", ticklen=5, zeroline=False),
            yaxis=dict(title="Netsolar", ticklen=5, zeroline=False),
        )

        fig = dict(data=data, layout=layout)
        iplot(fig)
    
    
    def roll_mean_plot(self, train_dict, roll_mean_d):    
        fig = go.Figure()
        
        for city in train_dict["city"].unique():
            fig.add_trace(go.Scatter(x=train_dict[train_dict["city"]==city].index,
                                     y=roll_mean_d[city]["netsolar"],
                                     name=city),
                        )

        fig.update_layout(
            title="<b>10 Days Rolling Average of Net Solar Radiation</b>",
            title_x=0.5, title_y=0.9,
            )
                    
        return fig.show()
    
    
    def pacf_plot(self, x_var, lags):
        fig, ax = plt.subplots(figsize=(13,3))
        plot_pacf(x_var, lags=lags, ax=ax)
        
        return plt.show()


    def acf_plot(self, x_var, lags):
        fig, ax = plt.subplots(figsize=(13,3))
        plot_acf(x_var, lags=lags, ax=ax)
        
        return plt.show()