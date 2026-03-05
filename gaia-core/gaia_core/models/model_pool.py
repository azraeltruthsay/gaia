from ._model_pool_impl import ModelPool
from gaia_core.config import get_config

config = get_config()
model_pool = ModelPool(config)

def get_model_pool():
    return model_pool