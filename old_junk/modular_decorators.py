from typing import Callable

NOISE_PREDICTION_MODULES = {} # type: dict[str, Callable]
POST_NOISE_GUIDANCE_MODULES = {} # type: dict[str, Callable]
PRE_NOISE_GUIDANCE_MODULES = {} # type: dict[str, Callable]

def module_noise_pred(name: str) -> Callable:
    """Register a noise prediction module to be used in the noise prediction pipeline.

    Args:
        name (str): name of the module
    """
    def wrapper(module):
        if name in NOISE_PREDICTION_MODULES:
            raise ValueError(f"Module with name {name} already registered.")
        NOISE_PREDICTION_MODULES[name] = module
        print(f"Registered module {name} as noise prediction module.")

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



def module_post_noise_guidance(name: str) -> Callable:
    """Register a post noise guidance module to be used in the post noise guidance pipeline.

    Args:
        name (str): name of the module
    """
    def wrapper(module):
        if name in POST_NOISE_GUIDANCE_MODULES:
            raise ValueError(f"Module with name {name} already registered.")
        POST_NOISE_GUIDANCE_MODULES[name] = module
        print(f"Registered module {name} as post noise guidance module.")

        return module
    return wrapper

def get_post_noise_guidance_module(name: str) -> Callable:
    """Get a post noise guidance module by name.

    Args:
        name (str): name of the module

    Raises:
        ValueError: if module with name not registered

    Returns:
        Callable: module
    """
    if name is None:
        return get_post_noise_guidance_module("default_case")

    if name not in POST_NOISE_GUIDANCE_MODULES:
        raise ValueError(f"Module with name {name} not registered.")
    return POST_NOISE_GUIDANCE_MODULES[name]



def module_pre_noise_guidance(name: str) -> Callable:
    """Register a pre noise guidance module to be used in the pre noise guidance pipeline.

    Args:
        name (str): name of the module
    """
    def wrapper(module):
        if name in PRE_NOISE_GUIDANCE_MODULES:
            raise ValueError(f"Module with name {name} already registered.")
        PRE_NOISE_GUIDANCE_MODULES[name] = module
        print(f"Registered module {name} as pre noise guidance module.")

        return module
    return wrapper

def get_pre_noise_guidance_module(name: str) -> Callable:
    """Get a pre noise guidance module by name.

    Args:
        name (str): name of the module

    Raises:
        ValueError: if module with name not registered

    Returns:
        Callable: module
    """
    if name is None:
        return get_pre_noise_guidance_module("default_case")

    if name not in PRE_NOISE_GUIDANCE_MODULES:
        raise ValueError(f"Module with name {name} not registered.")
    return PRE_NOISE_GUIDANCE_MODULES[name]
