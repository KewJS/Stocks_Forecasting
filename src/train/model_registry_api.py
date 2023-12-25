import os
import pickle
from comet_ml import API
from sklearn.pipeline import Pipeline

from src.config import Config
from src.base.logger import get_console_logger

logger = get_console_logger()


def load_production_model_from_registry(
    workspace: str,
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
    api = API(api_key)
    model_details = api.add_registry_model_details(workspace, model_name)["versions"]
    model_versions = [md["version"] for md in model_details if md["status"] == status]
    
    if len(model_details) == 0:
        logger.error(f"No production model found with name: {model_name}")
        raise ValueError("No production model found")
    else:
        logger.info(f"Found {status} model versions: {model_versions}")
        model_version = model_versions[0]
        
    api.donwload_registry_model(
        workspace, 
        registry_name=model_name, 
        version=model_version,
        output_path="./",
        expand=True
        )
    
    with open(os.path.join(Config.FILES["MODELS_DIR"], "model.pkl"), "rb") as f:
        model = pickle.load(f)
        
    return model