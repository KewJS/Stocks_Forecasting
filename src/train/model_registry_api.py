import os
import pickle
from comet_ml.api import API
from sklearn.pipeline import Pipeline

from src.config import Config
from src.base.logger import get_console_logger

logger = get_console_logger()


def load_production_model_from_registry(workspace: str,
                                        api_key: str,
                                        model_name: str,
                                        status: str="Production"
                                        ) -> Pipeline:
    """Loads the production model from the remote model registry

    Args:
        workspace (str): _description_
        api_key (str): _description_
        model_name (str): _description_
        status (str, optional): _description_. Defaults to "Production".

    Returns:
        Pipeline: _description_
    """
    api = API(api_key=api_key)
    model_details = api.get_registry_model_details(workspace, model_name)["versions"]
    model_versions = [md["version"] for md in model_details if md["status"].lower() == status]
    
    if len(model_details) == 0:
        logger.error(f"No production model found with name: {model_name}")
        raise ValueError("No production model found")
    else:
        logger.info(f"Found {status} model versions: {model_versions}")
        model_version = model_versions[0]
        
    api.download_registry_model(
        workspace, 
        registry_name=model_name, 
        version=model_version,
        output_path=os.path.join(Config.FILES["MODELS_DIR"], "stocks"),
        expand=True
        )
    
    if "lgb" in model_name:
        model_substr = "lightgbm"
    elif "xgb" in model_name:
        model_substr = "xgboost"
    elif "lasso" in model_name:
        model_substr = "lasso"
    else:
        raise ValueError(f"Unknown model name: {model_substr}")
    
    with open(os.path.join(Config.FILES["MODELS_DIR"], "stocks", "{}_model.pkl".format(model_substr)), "rb") as f:
        model = pickle.load(f)
        
    return model