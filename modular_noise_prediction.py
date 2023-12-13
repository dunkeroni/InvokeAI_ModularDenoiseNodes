from typing import Callable

NOISE_PREDICTION_MODULES = {} # type: dict[str, Callable]

def module_noise_pred(name: str) -> Callable:
    """Register a noise prediction module to be used in the noise prediction pipeline.

    Args:
        name (str): name of the module
    """
    def wrapper(module):
        if name in NOISE_PREDICTION_MODULES:
            raise ValueError(f"Module with name {name} already registered.")
        NOISE_PREDICTION_MODULES[name] = module

        return module
    return wrapper

def get_noise_prediction_module(name: str) -> Callable:
    """Get a noise prediction module by name.

    Args:
        name (str): name of the module

    Raises:
        ValueError: if module with name not registered

    Returns:
        Callable: module
    """
    if name is None:
        return get_noise_prediction_module("standard_unet_step_module") #default case

    if name not in NOISE_PREDICTION_MODULES:
        raise ValueError(f"Module with name {name} not registered.")
    return NOISE_PREDICTION_MODULES[name]