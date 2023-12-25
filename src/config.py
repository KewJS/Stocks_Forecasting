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
        CRYPTO_PRODUCT_ID = "BTC-USD",
        CRYPTO_FROM_DAY = "2020-01-01",
        CRYPTO_TO_DAY = "2023-11-01",
        
        STOCK_TICKER = "AAPL",
        ALPHA_VANTAGE_API = "GFOAU6X4ZT7H4BBH",
    )
    
    ANALYSIS_CONFIG = dict(
        # # crypto or stocks
        INPUT_SEQ_LEN       = 24,
        STEP_SIZE           = 1,
        
        # # radiation
        YEAR_LIST           = ["2017", "2018", "2019", "2020", "2021", "2022", "2023", "2023"], # "2017", "2018", 
        LOCATION_LIST       = ["Bondville_IL", "Boulder_CO", "Desert_Rock_NV", "Fort_Peck_MT", 
                               "Goodwin_Creek_MS", "Penn_State_PA", "Sioux_Falls_SD"],
        READ_MERGE          = True,
        DIFFERENCE_ORDER    = 1,
        TARGET_VAR          = "netsolar",
        SAMPLE_RATE         = "hourly", # daily, hourly
    )
    
    MODELLING_CONFIG = dict(
        UNIVARIATE_FORECAST     = False,
        SPLIT_RATIO             = 0.9,
        
        NUMBER_OF_LAGS          = 7,
        STANDARDIZATION_METHOD  = "minmax",
        DATA_BATCH_SIZE         = 16,
        FEATURE_ENGINEER        = True,
        NORMALIZATION_METHOD    = "", # "minmax"
        
        LAGS_WINDOW             = 50,
        OUTPUT_DIM              = 1,
        HIDDEN_DIM              = 64,
        LAYER_DIM               = 3,
        BATCH_SIZE              = 16,
        DROPOUT                 = 0.2,
        N_EPOCHS                = 20,
        LEARNING_RATE           = 1e-3,
        WEIGHT_DECAY            = 1e-6,
        
    )
    
    VARS = OrderedDict(
        STOCKS = [
            dict(var="time",    impute="",  predictive=True ),
            dict(var="open",    impute="",  predictive=False),
            dict(var="high",    impute="",  predictive=False),
            dict(var="low",     impute="",  predictive=False),
            dict(var="close",   impute="",  predictive=True),
            dict(var="volume",  impute="",  predictive=False),
        ],
        
        RADIATION = [
            dict(var="year",                    min=1,      max=206209 ,    impute="",  predictive=True ),
            dict(var="jday",                    min=1,      max=145    ,    impute="",  predictive=True ),
            dict(var="month",                   min=0,      max=1      ,    impute="",  predictive=True ),
            dict(var="day",                     min=0,      max=3421083,    impute="",  predictive=True ),
            dict(var="hour",                    min=None,   max=None   ,    impute="",  predictive=True ),
            dict(var="min",                     min=1,      max=99     ,    impute="",  predictive=True ),
            dict(var="dt",                      min=0,      max=6      ,    impute="",  predictive=True ),
            dict(var="solar_zenith",            min=0,      max=23     ,    impute="",  predictive=True ),
            dict(var="ghi",                     min=0.0,    max=30.0   ,    impute="",  predictive=True ),
            dict(var="ghi_flag",                min=0,      max=1,          impute="",  predictive=False),
            dict(var="uw_solar",                min=None,   max=None   ,    impute="",  predictive=True ),
            dict(var="uw_solar_flag",           min=0,      max=1,          impute="",  predictive=False),
            dict(var="dni",                     min=None,   max=None   ,    impute="",  predictive=True ),
            dict(var="dni_flag",                min=0,      max=1,          impute="",  predictive=False),
            dict(var="dhi",                     min=None,   max=None   ,    impute="",  predictive=True ), 
            dict(var="dhi_flag",                min=0,      max=1,          impute="",  predictive=False), 
            dict(var="dw_ir",                   min=None,   max=None   ,    impute="",  predictive=True ), 
            dict(var="dw_ir_flag",              min=0,      max=1,          impute="",  predictive=False), 
            dict(var="dw_casetemp",             min=None,   max=None   ,    impute="",  predictive=True ), 
            dict(var="dw_casetemp_flag",        min=0,      max=1,          impute="",  predictive=False), 
            dict(var="dw_dometemp",             min=None,   max=None   ,    impute="",  predictive=True ), 
            dict(var="dw_dometemp_flag",        min=0,      max=1,          impute="",  predictive=False), 
            dict(var="uw_ir",                   min=None,   max=None   ,    impute="",  predictive=True ), 
            dict(var="uw_ir_flag",              min=0,      max=1,          impute="",  predictive=False), 
            dict(var="uw_casetemp",             min=None,   max=None   ,    impute="",  predictive=True ),
            dict(var="uw_casetemp_flag",        min=0,      max=1,          impute="",  predictive=False),
            dict(var="uw_dometemp",             min=None,   max=None   ,    impute="",  predictive=True ),
            dict(var="uw_dometemp_flag",        min=0,      max=1,          impute="",  predictive=False),  
            dict(var="uvb",                     min=None,   max=None   ,    impute="",  predictive=True ),  
            dict(var="uvb_flag",                min=0,      max=1,          impute="",  predictive=False),  
            dict(var="par",                     min=None,   max=None   ,    impute="",  predictive=True ),  
            dict(var="par_flag",                min=0,      max=1,          impute="",  predictive=False),  
            dict(var="netsolar",                min=None,   max=None   ,    impute="",  predictive=False),  
            dict(var="netsolar_flag",           min=0,      max=1,          impute="",  predictive=False),  
            dict(var="netir",                   min=None,   max=None   ,    impute="",  predictive=True ),  
            dict(var="netir_flag",              min=0,      max=1,          impute="",  predictive=False),
            dict(var="totalnet",                min=None,   max=None   ,    impute="",  predictive=True ),
            dict(var="totalnet_flag",           min=0,      max=1,          impute="",  predictive=False),
            dict(var="temp_air",                min=None,   max=None   ,    impute="",  predictive=True ),
            dict(var="temp_air_flag",           min=0,      max=1,          impute="",  predictive=False),
            dict(var="relative_humidity",       min=None,   max=None   ,    impute="",  predictive=True ),
            dict(var="relative_humidity_flag",  min=0,      max=1,          impute="",  predictive=False),
            dict(var="wind_speed",              min=None,   max=None   ,    impute="",  predictive=True ),
            dict(var="wind_speed_flag",         min=0,      max=1,          impute="",  predictive=False),
            dict(var="wind_direction",          min=None,   max=None   ,    impute="",  predictive=True ),
            dict(var="wind_direction_flag",     min=0,      max=1,          impute="",  predictive=False),
            dict(var="pressure",                min=None,   max=None   ,    impute="",  predictive=True ),
            dict(var="pressure_flag",           min=0,      max=1,          impute="",  predictive=False),
            dict(var="city",                    min=None,   max=None   ,    impute="",  predictive=False),
        ],
        
        FEATURE_ENGINEER = [
            dict(var="*_lags*",                     impute="",  predictive=False),
            dict(var="day",                         impute="",  predictive=False),
            dict(var="sin_hour",                    impute="",  predictive=False),
            dict(var="cos_hour",                    impute="",  predictive=False),
            dict(var="sin_month",                   impute="",  predictive=False),
            dict(var="cos_month",                   impute="",  predictive=False),
            dict(var="sin_day_of_week",             impute="",  predictive=False),
            dict(var="cos_day_of_week",             impute="",  predictive=False),
            dict(var="is_holiday",                  impute="",  predictive=False),
            
            dict(var="rsi",                         impute="",  predictive=True ),
            dict(var="price_*_hour_ago",            impute="",  predictive=True ),
            dict(var="percentage_return_*_hour",    impute="",  predictive=True ),
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