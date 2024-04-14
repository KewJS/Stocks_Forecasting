import fnmatch
import pandas as pd
import os, sys, inspect
from datetime import datetime, timedelta
from collections import OrderedDict

base_path, currentdir = os.path.split(os.path.dirname(inspect.getfile(inspect.currentframe())))

class Config(object):

    QDEBUG = True

    NAME = dict(
        full = "time_series_forecasting",
    )

    FILES = dict(
        DATA_LOCAL              = "data",
        RAW_DATA                = os.path.join(base_path, "data", "raw_data"),
        PREPROCESS_DATA         = os.path.join(base_path, "data", "preprocess_data"),
        MODELS_DIR              = os.path.join(base_path, "data", "models"),
        
        PREPROCESS_CRYPTO_FILE  = "ohlc_data",
        PREPROCESS_STOCK_FILE   = "stock_data",
        
    )
    
    SCRAPER_CONFIG = dict(
        CRYPTO_PRODUCT_ID       = "BTC-USD",
        KLSE_COMPANY_LIST       = ["7155.KL", "7115.KL"],
        KLSE_COMPANY_NAME       = ["SKPRES", "SKBSHUT"],
        END_DATE                = datetime.now(),
        START_DATE_DIFF         = 8,
        ALPHA_VANTAGE_API       = "GFOAU6X4ZT7H4BBH",
    )
    
    ANALYSIS_CONFIG = dict(
        LAGS_WINDOW     = 7, # week period
        INPUT_SEQ_LEN   = 24,
        STEP_SIZE       = 1,
    )
    
    MODELLING_CONFIG = dict(
        TARGET_VAR          = "close",
        UNIVARIATE_FORECAST = False,
        SPLIT_RATIO         = 0.8,
        BEST_METRICS        = "RMSE", # MAE, MAPE, R2_Score, RMSE
        
        RANDOM_STATE        = 42,
        NUMBER_OF_SPLITS    = 5,
        EARLY_STOP_ROUND    = 10,
        
        # # OPTIMIZATION
        OPTIMIZED_DIRECTION = "minimize",
        NUM_TRIALS          = 20,
        
        # # LSTM
        INPUT_DIM           = 1,
        LAYER_DIM           = 2,
        HIDDEN_DIM          = 32,
        OUTPUT_DIM          = 1,
        DROPOUT_PROB        = 0.2,
        
        # # TRAINING
        LOOKBACK_SEQ        = 7,
        BATCH_SIZE          = 16,
        NUM_EPOCHS          = 100,
        LEARNING_RATE       = 0.001,
        WEIGHT_DECAY        = 1e-3,
        EPS                 = 1e-9,
        SCHEDULER_STEP_SIZE = 40,
        GAMMA               = 0.1,
    )
    
    VARS = OrderedDict(
        STOCKS = [
            dict(var="date",        impute="",  predictive=True ),
            dict(var="open",        impute="",  predictive=False),
            dict(var="high",        impute="",  predictive=False),
            dict(var="low",         impute="",  predictive=False),
            dict(var="close",       impute="",  predictive=True),
            dict(var="adj_close",   impute="",  predictive=False),
            dict(var="volume",      impute="",  predictive=False),
            dict(var="symbol",      impute="",  predictive=False),
            dict(var="name",        impute="",  predictive=False),
        ],
        
        FEATURE_ENGINEER = [
            dict(var="*_lag_t*",                    impute="",  predictive=True),
            dict(var="day",                         impute="",  predictive=False),
            dict(var="sin_*",                       impute="",  predictive=False),
            dict(var="cos_*",                       impute="",  predictive=False),
            dict(var="is_holiday",                  impute="",  predictive=False),
            dict(var="rsi",                         impute="",  predictive=False),
            dict(var="price_*_hour_ago",            impute="",  predictive=False),
            dict(var="percentage_return_*_hour",    impute="",  predictive=False),
        ]
    )
    
    
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