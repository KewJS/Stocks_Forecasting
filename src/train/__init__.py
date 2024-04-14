
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({"figure.max_open_warning": 0})
plt.style.use("fivethirtyeight")
cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm
from typing import Callable

import torch
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from src.config import Config
from src.train.hyperparams import Optimization
from src.models import REGRESSION_ALGORITHMS


class Logger(object):
    info     = print
    warning  = print
    error    = print
    critical = print
    

class Train(Config):
    data = {}
    
    def __init__(self, model=Callable, loss_fn=None, optimizer=None, logger=Logger()):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.logger = logger
        self.opt = Optimization()
        
        
    def train_pipeline(self):
        self._get_model_data()
        
        # base_result, base_metrics = self.build_baseline_model(self.data["model_data"].drop("close", axis=1), 
        #                                                       self.data["model_data"]["close"])
    
    
    def _get_model_data(self):
        fname = os.path.join(self.FILES["PREPROCESS_DATA"], "stocks", "{}.parquet".format(self.FILES["PREPROCESS_STOCK_FILE"]))
        self.data["model_data"] = pd.read_parquet(fname)
        
        
    def training_pipeline(self, X: pd.DataFrame, y: pd.Series):
        tscv = TimeSeriesSplit(n_splits=self.MODELLING_CONFIG["NUMBER_OF_SPLITS"])

        self.tuned_models = {}
        full_actual = []
        full_y_pred = []
        results_df = pd.DataFrame()

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
            
            for model_name, model_params in REGRESSION_ALGORITHMS.items():
                if model_name == "LSTM":
                    y_pred, actual, seq_results = self.opt.ts_dl_train(X_train, 
                                                                       y_train, 
                                                                       X_train.index.values, 
                                                                       model_name, 
                                                                       model_params)
                    full_actual.append(actual)
                    full_y_pred.append(y_pred)
                    results_df = pd.concat([results_df, seq_results], axis=0)
                    
                    results_df["actual"] = results_df["actual"].apply(lambda x: x[0])
                    results_df["prediction"] = results_df["prediction"].apply(lambda x: x[0])
                    results_df["date"] = pd.to_datetime(results_df["date"])
                    metrics = self.opt.evaluate(results_df["actual"], results_df["prediction"])
                    self.tuned_models[model_name] = {"model": model_name, "params": model_params["args"], "metrics": metrics}
                else:
                    model = self.opt.get_model(model_name, model_params)
                    
                    def objective(trial):
                        updated_params = {param_name: param_range(trial) 
                                        for param_name, param_range in model_params["param_grid"].items()}
                        model.set_params(**updated_params)
                        
                        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        
                        return mse
                    
                    study = optuna.create_study(direction=self.MODELLING_CONFIG["OPTIMIZED_DIRECTION"])
                    study.optimize(objective, n_trials=self.MODELLING_CONFIG["NUM_TRIALS"])
                    
                    best_model_params = model_params.copy()
                    best_model_params["params"] = study.best_params
                    best_model = model_params["alg"](**model_params["args"])
                    best_model.set_params(**study.best_params)
                    best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                    y_pred = best_model.predict(X_test)
                    metrics = self.opt.evaluate(y_test, y_pred)
                    self.tuned_models[model_name] = {"model": best_model, 
                                                     "params": study.best_params, 
                                                     "metrics": metrics}
                
        self.metrics_df, self.best_model = self.opt.get_results(self.tuned_models)
    
    
    def select_features(self, df: pd.DataFrame, target: str, correlation_type: str, threshold: int|float) -> pd.DataFrame:
        """Create a new DataFrame with only the features that have a correlation with the target above the threshold.

        Args:
            df (pd.DataFrame): Input dataframe
            target (str): Target variable
            correlation_type (str): Correlation type with "target" col
            threshold (int | float): Threshold value for correlation

        Raises:
            SystemError: If threshold is not between -1 and 1

        Returns:
            pd.DataFrame: Output dataframe with only the features that have a correlation with the target above the threshold
        """
        if (threshold < -1) | (threshold > 1):
            raise SystemError("correlation threshold must be between -1 and 1")
        
        features = df.corr(correlation_type).loc[target].drop(target)
        best_features = features.where(abs(features) > threshold).dropna()
        df = pd.concat([df[best_features.index], df[target]], axis=1)
        
        return df
    
    
    def prepare_uni_ts_x(self, 
                         x: pd.DataFrame, 
                         window_size: int=Config.MODELLING_CONFIG["LOOKBACK_SEQ"], 
                         univariate: bool=Config.MODELLING_CONFIG["UNIVARIATE_FORECAST"]):
        """Convert univariate data into sequences of data, based on the window size defined
        
        etc.
        data_x, data_x_unseen = prepare_ts_x(X, 7)

        Args:
            x (_type_): _description_
            window_size (_type_): _description_
            univariate (_type_, optional): _description_. Defaults to Config.MODELLING_CONFIG["UNIVARIATE_FORECAST"].

        Returns:
            _type_: _description_
        """
        x = x.values
        n_row = x.shape[0] - window_size + 1
        output = np.lib.stride_tricks.as_strided(x, shape=(n_row,window_size), strides=(x.strides[0],x.strides[0]))
        
        return output[:-1], output[-1]


    def prepare_uni_ts_y(x, window_size):
        # # perform simple moving average
        # output = np.convolve(x, np.ones(window_size), 'valid') / window_size
        # use the next day as label
        output = x[window_size:]
        
        return output
    
    
    def plot_cv_indices(self, cv, n_splits, X, y, date_col = None):
        """Create a sample plot for indices of a cross-validation object."""
        fig, ax = plt.subplots(1, 1, figsize = (11, 7))
        for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
            indices = np.array([np.nan] * len(X))
            indices[tt] = 1
            indices[tr] = 0
            ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                    c=indices, marker="_", lw=10, cmap=cmap_cv,
                    vmin=-.2, vmax=1.2)
        yticklabels = list(range(n_splits))
        
        if date_col is not None:
            tick_locations  = ax.get_xticks()
            tick_dates = [" "] + date_col.iloc[list(tick_locations[1:-1])].astype(str).tolist() + [" "]

            tick_locations_str = [str(int(i)) for i in tick_locations]
            new_labels = ["\n\n".join(x) for x in zip(list(tick_locations_str), tick_dates) ]
            ax.set_xticks(tick_locations)
            ax.set_xticklabels(new_labels)
        
        ax.set(yticks=np.arange(n_splits) + .5, yticklabels=yticklabels,
            xlabel="Sample index", ylabel="CV iteration",
            ylim=[n_splits+0.2, -.2])
        ax.legend([Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.02))],
                ["Testing set", "Training set"], loc=(1.02, .8))
        ax.set_title("{}".format(type(cv).__name__), fontsize=15)
        
        return fig
    
    
class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu