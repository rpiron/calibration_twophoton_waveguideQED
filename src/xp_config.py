from dataclasses import dataclass
from typing import Dict, Optional, Any
from src.rk_integrator import rk_propagator
import numpy as np

@dataclass
class ExperimentConfig:
    """
    Configuration class for Experiment parameters.
    """

    param_photons: Dict[str, np.ndarray]
    param_cavity: Dict[str, float]
    param_time_evol: Dict[str, float]
    cutoffs: Dict[str, float]
    integrator_func: Optional[Any] = rk_propagator
    store_state: bool = True