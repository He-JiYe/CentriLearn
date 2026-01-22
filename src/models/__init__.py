"""
Architecture:
- __init__.py: Model building utilities.
- nn/: Custom Neural network (GraphSAGE, etc.)
- backbones/: Backbone networks (SimpleNet, DeepNet, FPNet)
- network_dismantling/: ND-specific models (Qnet, ActorCritic)
- utils/: Model building utilities and model registration.
"""
from .utils.builder import build_network_dismantler
from .utils.registry import NETWORK_DISMANTLER

__all__ = [
    'build_network_dismantler', 
    'NETWORK_DISMANTLER'
]
