"""Methods Registry for plug-and-play method selection."""
from typing import Type, Dict
from base.base import AbsMethod

# Import method classes
from .asif import ASIFMethod
from .csa import CSAMethod
from .cka import CKAMethod

_METHOD_REGISTRY: Dict[str, Type[AbsMethod]] = {
    "asif": ASIFMethod,
    "csa": CSAMethod,
    "cka": CKAMethod,
}


def get_method_class(name: str) -> Type[AbsMethod]:
    """
    Retrieve a method class by name.
    
    Args:
        name: The name of the method to retrieve (e.g., "asif", "csa").
        
    Returns:
        The method class.
        
    Raises:
        ValueError: If the method name is not found in the registry.
    """
    method_class = _METHOD_REGISTRY.get(name.lower())
    if method_class is None:
        available = list(_METHOD_REGISTRY.keys())
        raise ValueError(f"Method '{name}' not found. Available methods: {available}")
    return method_class


def list_methods() -> list[str]:
    """List all available registered methods."""
    return list(_METHOD_REGISTRY.keys())
