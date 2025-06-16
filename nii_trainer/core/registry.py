"""
Component registry system for NII-Trainer.

This module provides a centralized way to register and retrieve components
like models, optimizers, schedulers, etc.
"""

from typing import Dict, Type, Any, Callable
from .exceptions import ConfigurationError

class Registry:
    """Registry for storing and retrieving components."""
    
    def __init__(self, name: str):
        self.name = name
        self.registry: Dict[str, Any] = {}
    
    def register(self, name: str, component: Any) -> None:
        """Register a component with the given name."""
        if name in self.registry:
            raise ConfigurationError(f"Component '{name}' already registered in {self.name}")
        self.registry[name] = component
    
    def get(self, name: str) -> Any:
        """Get a component by name."""
        if name not in self.registry:
            raise ConfigurationError(f"Component '{name}' not found in {self.name}")
        return self.registry[name]
    
    def has(self, name: str) -> bool:
        """Check if a component is registered."""
        return name in self.registry
    
    def list_components(self) -> list:
        """List all registered component names."""
        return list(self.registry.keys())
    
    def decorator(self, name: str = None):
        """Decorator for registering components."""
        def _decorator(component):
            component_name = name or component.__name__
            self.register(component_name, component)
            return component
        return _decorator


class ComponentRegistry:
    """Alias for Registry class for backward compatibility."""
    
    def __init__(self, name: str):
        self.name = name
        self.registry: Dict[str, Any] = {}
    
    def register(self, name: str, component: Any) -> None:
        """Register a component with the given name."""
        if name in self.registry:
            raise ConfigurationError(f"Component '{name}' already registered in {self.name}")
        self.registry[name] = component
    
    def get(self, name: str) -> Any:
        """Get a component by name."""
        if name not in self.registry:
            raise ConfigurationError(f"Component '{name}' not found in {self.name}")
        return self.registry[name]
    
    def has(self, name: str) -> bool:
        """Check if a component is registered."""
        return name in self.registry
    
    def list_components(self) -> list:
        """List all registered component names."""
        return list(self.registry.keys())


# Global registries
MODELS = Registry("models")
ENCODERS = Registry("encoders") 
DECODERS = Registry("decoders")
LOSSES = Registry("losses")
OPTIMIZERS = Registry("optimizers")
SCHEDULERS = Registry("schedulers")
TRANSFORMS = Registry("transforms")
DATASETS = Registry("datasets")
TRAINERS = Registry("trainers")
EVALUATORS = Registry("evaluators")

# Convenience functions
def register_model(name: str = None):
    """Decorator to register a model."""
    return MODELS.decorator(name)

def register_encoder(name: str = None):
    """Decorator to register an encoder."""
    return ENCODERS.decorator(name)

def register_decoder(name: str = None):
    """Decorator to register a decoder."""
    return DECODERS.decorator(name)

def register_loss(name: str = None):
    """Decorator to register a loss function."""
    return LOSSES.decorator(name)

def register_optimizer(name: str = None):
    """Decorator to register an optimizer."""
    return OPTIMIZERS.decorator(name)

def register_scheduler(name: str = None):
    """Decorator to register a scheduler."""
    return SCHEDULERS.decorator(name)

def register_transform(name: str = None):
    """Decorator to register a transform."""
    return TRANSFORMS.decorator(name)

def register_dataset(name: str = None):
    """Decorator to register a dataset."""
    return DATASETS.decorator(name)

def register_trainer(name: str = None):
    """Decorator to register a trainer."""
    return TRAINERS.decorator(name)

def register_evaluator(name: str = None):
    """Decorator to register an evaluator."""
    return EVALUATORS.decorator(name)