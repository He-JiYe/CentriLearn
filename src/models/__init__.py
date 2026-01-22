"""
Models module for graph neural networks.

This module provides a flexible, registry-based system for building
graph neural networks and task-specific models.
"""
from .utils.builder import build_backbone, build_head, build_network_dismantler
from .utils.registry import BACKBONES, HEADS, NETWORK_DISMANTLER

__all__ = [
    'build_backbone',
    'build_head',
    'build_network_dismantler',
    'BACKBONES',
    'HEADS',
    'NETWORK_DISMANTLER'
]
