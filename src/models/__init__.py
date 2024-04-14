import optuna
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


from src.config import Config
from src.models.rnn import RNNModel
from src.models.lstm import LSTMModel
from src.models.gru import GRUModel


trial = optuna.trial.Trial
REGRESSION_ALGORITHMS = dict(  
    XGBR_tuned = dict(
        alg=XGBRegressor, 
        args=dict(
            silent=0, 
            verbosity=0,
            random_state=Config.MODELLING_CONFIG["RANDOM_STATE"], 
            objective="reg:squarederror",
            early_stopping_rounds=Config.MODELLING_CONFIG["EARLY_STOP_ROUND"]), 
        scaled=False, 
        param_grid={
            "max_leaves"        : lambda trial: trial.suggest_int(name="max_leaves", low=2, high=256),
            "eta"               : lambda trial: trial.suggest_float(name="eta", low=0.2, high=1.0),
            "gamma"             : lambda trial: trial.suggest_float(name="gamma", low=0.2, high=1.0),
            "max_depth"         : lambda trial: trial.suggest_int(name="max_depth", low=3, high=50),
            "subsample"         : lambda trial: trial.suggest_float(name="subsample", low=0.2, high=1.0),
            "lambda"            : lambda trial: trial.suggest_int(name="lambda", low=1, high=20),
            "alpha"             : lambda trial: trial.suggest_int(name="alpha", low=1, high=20),
            "n_estimators"      : lambda trial: trial.suggest_int(name="n_estimators", low=20, high=500),
            "colsample_bytree"  : lambda trial: trial.suggest_float(name="colsample_bytree", low=0.2, high=1.0),
            },
        ),
    LGBMR_tuned = dict(
        alg=LGBMRegressor, 
        args=dict(
            random_state=Config.MODELLING_CONFIG["RANDOM_STATE"],
            objective="regression",
            force_col_wise=True), 
        scaled=False, 
        param_grid={
            "learning_rate"     : lambda trial: trial.suggest_float("learning_rate", low=0.2, high=1.0),
            "n_estimators"      : lambda trial: trial.suggest_int("n_estimators", 20, 500),
            "num_leaves"        : lambda trial: trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction"  : lambda trial: trial.suggest_float("feature_fraction", 0.2, 1.0),
            "bagging_fraction"  : lambda trial: trial.suggest_float("bagging_fraction", 0.2, 1.0),
            "min_child_samples" : lambda trial: trial.suggest_int("min_child_samples", 3, 100),
            }, 
        ),
    LSTM = dict(
        alg=LSTMModel,
        args=dict(
            input_dim=Config.MODELLING_CONFIG["INPUT_DIM"],
            layer_dim=Config.MODELLING_CONFIG["LAYER_DIM"],
            hidden_dim=Config.MODELLING_CONFIG["HIDDEN_DIM"],
            output_dim=Config.MODELLING_CONFIG["OUTPUT_DIM"],
            dropout_prob=Config.MODELLING_CONFIG["DROPOUT_PROB"]
        ),
        scaled=False,
        param_grid={}
        ),
    RNN = dict(
        alg=RNNModel,
        args=dict(
            input_dim=Config.MODELLING_CONFIG["INPUT_DIM"],
            layer_dim=Config.MODELLING_CONFIG["LAYER_DIM"],
            hidden_dim=Config.MODELLING_CONFIG["HIDDEN_DIM"],
            output_dim=Config.MODELLING_CONFIG["OUTPUT_DIM"],
            dropout_prob=Config.MODELLING_CONFIG["DROPOUT_PROB"]
        ),
        scaled=False,
        param_grid={}
        ),
    GRU = dict(
        alg=GRUModel,
        args=dict(
            input_dim=Config.MODELLING_CONFIG["INPUT_DIM"],
            layer_dim=Config.MODELLING_CONFIG["LAYER_DIM"],
            hidden_dim=Config.MODELLING_CONFIG["HIDDEN_DIM"],
            output_dim=Config.MODELLING_CONFIG["OUTPUT_DIM"],
            dropout_prob=Config.MODELLING_CONFIG["DROPOUT_PROB"]
        ),
        scaled=False,
        param_grid={}
        )
    )
