import os
import math
import numpy as np
import pandas as pd
from comet_ml import Experiment
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Callable, Union

import optuna
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from src.analysis.preprocessing import get_preprocessing_pipeline
from src.base.logger import get_console_logger
from src.config import Config

logger = get_console_logger()


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, dates):
        self.X = X
        self.y = y
        self.dates = dates.astype(str)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.dates[idx]


class Optimization(Config):
    def __init__(self, model=Callable, loss_fn=None, optimizer=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        
    def get_model(self, model_name, model_params):
        alg = model_params["alg"]
        args = model_params["args"]
        scaled = model_params.get("scaled", False)
        param_grid = model_params.get("param_grid", {})
        
        models = {
            model_name: alg
        }
        
        return models.get(model_name)(**args)


    def ts_data_generators(self, X, y, dates, batch_size=32, lookback=7):
        X = torch.tensor(X.values).float()
        y = torch.tensor(y.values).float()
        dates = dates.values if isinstance(dates, pd.Index) else dates
            
        train_size = int(len(X) * self.MODELLING_CONFIG["SPLIT_RATIO"] - (self.MODELLING_CONFIG["SPLIT_RATIO"] * 0.2))
        val_size = int(len(X) * (self.MODELLING_CONFIG["SPLIT_RATIO"] * 0.2))
        
        print("Crete train, validation & test dataset...")
        X_train = X[:train_size].reshape((-1, lookback, 1))
        X_val = X[train_size:train_size + val_size].reshape((-1, lookback, 1))
        X_test = X[train_size + val_size:].reshape((-1, lookback, 1))
        
        y_train = y[:train_size].reshape(-1, 1)
        y_val = y[train_size:train_size + val_size].reshape(-1, 1)
        y_test = y[train_size + val_size:].reshape(-1, 1)
        
        dates_train = dates[:train_size]
        dates_val = dates[train_size:train_size + val_size]
        dates_test = dates[train_size + val_size:]
        
        print("Convert tensors to DataLoader...")
        train_dataset = TimeSeriesDataset(X_train, y_train, dates_train)
        val_dataset = TimeSeriesDataset(X_val, y_val, dates_val)
        test_dataset = TimeSeriesDataset(X_test, y_test, dates_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader_one = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)
        
        print("Done creating train, validation & test tensors...")
        
        return train_loader, val_loader, test_loader, test_loader_one
     
        
    def train_step(self, x, y):
        """The method train_step completes one step of training.

        Given the features (x) and the target values (y) tensors, the method completes
        one step of the training. First, it activates the train mode to enable back prop.
        After generating predicted values (yhat) by doing forward propagation, it calculates
        the losses by using the loss function. Then, it computes the gradients by doing
        back propagation and updates the weights by calling step() function.

        Args:
            x (torch.Tensor): Tensor for features to train one step
            y (torch.Tensor): Tensor for target values to calculate losses

        """
        self.model.train()
        self.optimizer.zero_grad()

        yhat = self.model(x)
        loss = self.loss_fn(y, yhat)

        loss.backward()
        self.optimizer.step()
        
        return loss.item()


    def ts_dl_train(self, model_name, train_loader, val_loader, test_loader, batch_size=64, n_epochs=50, n_features=1):
        """The method train performs the model training

        The method takes DataLoaders for training and validation datasets, batch size for
        mini-batch training, number of epochs to train, and number of features as inputs.
        Then, it carries out the training by iteratively calling the method train_step for
        n_epochs times. If early stopping is enabled, then it  checks the stopping condition
        to decide whether the training needs to halt before n_epochs steps. Finally, it saves
        the model in a designated file path.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader that stores training data
            val_loader (torch.utils.data.DataLoader): DataLoader that stores validation data
            batch_size (int): Batch size for mini-batch training
            n_epochs (int): Number of epochs, i.e., train steps, to train
            n_features (int): Number of feature columns

        """
        model_dir = os.path.join(self.FILES["MODELS_DIR"], "stocks")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "{}_{}.pth".format(model_name, datetime.now().strftime("%Y-%m-%d %H-%M-%S")))

        test_pred = []
        test_actual = []
        test_dates = []

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch, _ in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)

            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val, _ in val_loader:
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) or (epoch % 50 == 0):
                print(f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}")

        torch.save(self.model.state_dict(), model_path)
        
        self.model.eval()
        seq_pred = []
        seq_actual = []
        seq_dates = []
        with torch.no_grad():
            for x_test, y_test, dates in test_loader:
                x_test, y_test = x_test.to(device), y_test.to(device)
                output = self.model(x_test)
                seq_pred.extend(output.cpu().numpy())
                seq_actual.extend(y_test.cpu().numpy())
                seq_dates.extend(dates)

        test_pred.extend(seq_pred)
        test_actual.extend(seq_actual)
        test_dates.extend(seq_dates)
        
        results_df = pd.DataFrame({
                "actual": test_actual,
                "prediction": test_pred,
                "date": test_dates
            })
        
        return results_df


    def evaluate(self, test_loader, batch_size=1, n_features=1):
        """The method evaluate performs the model evaluation

        The method takes DataLoaders for the test dataset, batch size for mini-batch testing,
        and number of features as inputs. Similar to the model validation, it iteratively
        predicts the target values and calculates losses. Then, it returns two lists that
        hold the predictions and the actual values.

        Note:
            This method assumes that the prediction from the previous step is available at
            the time of the prediction, and only does one-step prediction into the future.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader that stores test data
            batch_size (int): Batch size for mini-batch training
            n_features (int): Number of feature columns

        Returns:
            list[float]: The values predicted by the model
            list[float]: The actual values in the test set.

        """
        with torch.no_grad():
            pred = []
            actual = []
            dates = []
            for x_test, y_test, date in test_loader:
                # x_test = x_test.view([batch_size, -1, n_features]).to(self.device)
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)
                self.model.eval()
                
                yhat = self.model(x_test)
                yhat = yhat.cpu().data.numpy()
                pred.append(yhat)
                
                y_actual = y_test.cpu().data.numpy()
                actual.append(y_actual)
                
                dates.append(date)

        return pred, actual, dates


    def plot_losses(self):
        """The method plots the calculated loss values for training and validation
        """
        plt.style.use("ggplot")
        plt.figure(figsize=(10,5))
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()
        
        
    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred):
        mape = np.mean(np.abs((y_true - y_pred) / y_true+1e-6)) * 100
        if type(mape) == pd.Series: mape = mape[0]
        return mape


    @staticmethod
    def root_mean_square_error(y_true, y_pred):
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        return rmse


    @staticmethod
    def adjusted_r2_score(y_true, y_pred, k):
        r2score = r2_score(y_true, y_pred)
        n = len(y_true)
        if r2score < 0.3:
            return r2score
        adjusted_r2 = 1 - (1 - r2score) * (n - 1) / (n - k - 1)
        
        return adjusted_r2


    def evaluate(self, actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:        
        r2score = r2_score(actual, pred)
        MAPE = self.mean_absolute_percentage_error(actual, pred)
        MAE = mean_absolute_error(actual, pred)
        # r2score_adjusted = adjusted_r2_score(actual, pred, len(self.predictives))
        rmse = self.root_mean_square_error(actual, pred)

        metrics = dict(MAE=MAE, MAPE=MAPE, R2_Score=r2score, RMSE=rmse) # , R2_Score_Adjusted=r2score_adjusted

        return metrics


    def get_results(self, tuned_models_results: dict[str, dict]) -> dict[str, float]:
        metrics_data = {}
        best_model = {}
        for model_name, model_info in tuned_models_results.items():
            metrics_data[model_name] = model_info["metrics"]
            best_metrics = model_info["metrics"][self.MODELLING_CONFIG["BEST_METRICS"]]
            best_model[model_name] = best_metrics

        metrics_df = pd.DataFrame(metrics_data).T
        
        if self.MODELLING_CONFIG["BEST_METRICS"] == "R2_Score":
            best_model_name = max(best_model, key=best_model.get)
        else:
            best_model_name = min(best_model, key=best_model.get)
        best_model = tuned_models_results[best_model_name]
        
        print(f"""
            Best performing model: 
            {best_model_name}, {self.MODELLING_CONFIG['BEST_METRICS']}: {best_model['metrics'][self.MODELLING_CONFIG['BEST_METRICS']]}
            """)
        
        return metrics_df, best_model


def sample_hyperparam(model_fn: Callable,
                      trial: optuna.trial.Trial, 
                      ) -> Dict[str, Union[str, int, float]]:
    """Create the sample hyperparameters space for the model

    Args:
        model_fn (Callable): _description_
        trial (optuna.trial.Trial): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        Dict[str, Union[str, int, float]]: _description_
    """
    if model_fn == Lasso:
        return {
            "alpha": trial.suggest_float("alpha", 0.01, 1.0, log=True)
        }
    elif model_fn == XGBRegressor:
        return {
            "objective": "reg:squarederror",
            "verbosity": 0,
            "max_leaves": trial.suggest_int("max_leaves", 2, 256),
            "eta": trial.suggest_float("eta", 0.2, 1.0),
            "gamma": trial.suggest_float("gamma", 0.2, 1.0),
            "max_depth": trial.suggest_int("max_depth", 3, 50),
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            "lambda": trial.suggest_int("lambda", 1, 20),
            "alpha": trial.suggest_int("alpha", 1, 20),
        }
    elif model_fn == LGBMRegressor:
        return {
            "metric": "mae",
            "verbose": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.2, 1.0),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 100),
        }
    else:
        raise NotImplementedError("TODO: implement other models")
    
    
def find_best_hyperparams(model_fn: Callable,
                          hyperparam_trials: int,
                          x: pd.DataFrame,
                          y: pd.Series,
                          experiment: Experiment,
                          ) -> Tuple[Dict, Dict]:
    assert model_fn in {Lasso, XGBRegressor, LGBMRegressor}
    
    def objective(trial: optuna.trial.Trial) -> float:
        """Error function we want to minimize (or maximize) using hyperparameter tuning.

        Args:
            trial (optuna.trial.Trial): Trial object to store intermediate results of the optimization.

        Returns:
            float: _description_
        """
        preprocessing_hyperparams = {
            "pp_rsi_window": trial.suggest_int("pp_rsi_window", 5, 20),
        }
        model_hyperparams = sample_hyperparam(model_fn, trial)
        
        tss = TimeSeriesSplit(n_splits=3)
        scores = []
        logger.info(f"{trial.number=}")
        for split_number, (train_index, val_index) in enumerate(tss.split(x)):
            X_train, X_val = x.iloc[train_index], x.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            
            logger.info(f"{split_number=}")
            logger.info(f"{len(X_train)=}")
            logger.info(f"{len(X_val)=}")
            
            pipeline = make_pipeline(
                get_preprocessing_pipeline(**preprocessing_hyperparams),
                model_fn(**model_hyperparams)
            )
            pipeline.fit(X_train, y_train)  
            
            y_pred = pipeline.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            scores.append(mae)
            
            logger.info(f'{mae=}')
            
        score = np.array(scores).mean()
        
        return score
    
    logger.info("Starting hyperparameter search...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=hyperparam_trials)
    
    best_params = study.best_params
    best_value = study.best_value
    
    best_preprocessing_hyperparams = {
        key: value for key, value in best_params.items()
        if key.startswith("pp_")
    }
    
    best_model_hyperparams = {
        key: value for key, value in best_params.items()
        if not key.startswith("pp_")
    }
    
    logger.info("Best Parameters:")
    for key, value in best_params.items():
        logger.info(f"{key}: {value}")
    logger.info(f"Best MAE: {best_value}")
    
    experiment.log_metric("Cross Validation MAE", best_value)
    
    return best_preprocessing_hyperparams, best_model_hyperparams


class DL_Optimization:
    """Optimization is a helper class that allows training, validation, prediction.
    
    Optimization is a helper class that takes model, loss function, optimizer function
    learning scheduler (optional), early stopping (optional) as inputs. In return, it
    provides a framework to train and validate the models, and to predict future values
    based on the models.
    
    Attributes
    ----------
    model : RNNModel, LSTMModel, GRUModel
        Model class created for the type of RNN
    loss_fn : torch.nn.modules.Loss
        Loss function to calculate the losses
    optimizer  : torch.optim.Optimizer
        Optimizer function to optimize the loss function
    train_losses : List[float]
        The loss values from the training
    val_losses : List[float]
        The loss values from the validation
    last_epoch : int
        The number of epochs that the models is trained
    """
    def __init__(self, model, loss_fn, optimizer):
        """_summary_

        Parameters
        ----------
        model : RNNModel, LSTMModel, GRUModel
            Model class created for the type of RNN
        loss_fn : torch.nn.modules.Loss
            Loss function to calculate the losses
        optimizer : torch.optim.Optimizer
            Optimizer function to optimize the loss function
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        
        
    def train_step(self, x:torch.Tensor, y:torch.Tensor):
        """The method train_step completes one step of training
        
        Given the features (x) and the target values (y) tensors, the method
        completes one step of the training. First, it activates the train mode 
        to enable back prop. After generating predicted values (yhat) by doing 
        forward propagation, it calculates the losses by using the loss function. Then,
        it computes the gradients by doing back propagation and updates the weights 
        by calling step() function

        Parameters
        ----------
        x : torch.Tensor
            Tensor for features to train one step
        y : torch.Tensor
            Tensor for target values to calculate losses

        Returns
        -------
        _type_
            _description_
        """
        self.model.train()
        
        yhat = self.model(x)
        
        loss = self.loss_fn(y, yhat)
        
        loss.backward()
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    
    def train(self, train_loader:torch.utils.data.DataLoader, 
              val_loader:torch.utils.data.DataLoader, 
              batch_size:int=64, 
              n_epochs:int=40, 
              n_features:int=1):
        """The method train performs the model training
        
        The method takes DataLoader for training and validation datasets, batch size
        for mini-batch training, number of epochs to train and number of features as
        input. Then, it carries out the training by iteratively calling the method 
        train_step for n_epochs times. If early stopping is enabled, then it checks the
        stopping condition to decide whether the training needs to halt before n_epochs
        steps. Finally, it saves the model in a designed file path.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            Dataloader that stores training data
        val_loader : torch.utils.data.DataLoader
            Dataloader that stores validation data
        batch_size : int, optional
            Batch size for mini-batch training, by default 64
        n_epochs : int, optional
            Number of epochs, by default 40
        n_features : int, optional
            Number of feature columns, by default 1
        """
        model_path = f'{self.model}_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
            
            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

        torch.save(self.model.state_dict(), model_path)
        
        
    def evaluate(self, test_loader:torch.utils.data.DataLoader, batch_size:int=1, n_features:int=1):
        """The method evaluate performs the model evaluation
        
         The method takes DataLoaders for the test dataset, batch size for mini-batch testing,
        and number of features as inputs. Similar to the model validation, it iteratively
        predicts the target values and calculates losses. Then, it returns two lists that
        hold the predictions and the actual values.

        Parameters
        ----------
        test_loader : torch.utils.data.DataLoader
            DataLoader that stores test data
        batch_size : int, optional
            Batch size for mini-batch training, by default 1
        n_features : int, optional
            Number of feature columns, by default 1
            
        Returns:
        --------
        predictions : List[float]
            The values of predicted by model
        values : List[float]
            The actual values in the test set
        """
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                yhat=yhat.cpu().data.numpy()
                predictions.append(yhat)
                y_test=y_test.cpu().data.numpy()
                values.append(y_test)

        return predictions, values
    
    def plot_losses(self):
        plt.style.use('ggplot')
        plt.figure(figsize=(10,5))
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()