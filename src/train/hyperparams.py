import os
import numpy as np
import pandas as pd
from comet_ml import Experiment
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Callable, Union

import optuna
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from src.analysis.preprocessing import get_preprocessing_pipeline
from src.base.logger import get_console_logger

logger = get_console_logger()


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
    elif model_fn == LGBMRegressor:
        return {
            "metric": "mae",
            "verbose": -1,
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
    assert model_fn in {Lasso, LGBMRegressor}
    
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
            
            pipeline = get_preprocessing_pipeline(
                **preprocessing_hyperparams,
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