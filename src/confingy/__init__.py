"""
Confingy - A configuration management library with lazy instantiation and tracking.

This library provides decorators and utilities for:
- Lazy instantiation of classes (@lazy decorator)
- Tracking constructor arguments (@track decorator)
- Serialization/deserialization of configurations
"""

import logging

from confingy.exceptions import (
    DeserializationError,
    SerializationError,
    ValidationError,
)
from confingy.fingy import (
    deserialize_fingy,
    load_fingy,
    prettify_fingy,
    prettify_serialized_fingy,
    save_fingy,
    serialize_fingy,
    transpile_fingy,
)
from confingy.tracking import (
    Lazy,
    MaybeLazy,
    disable_validation,
    lazy,
    lens,
    track,
    update,
)

# Set up logging for better debugging
logger = logging.getLogger(__name__)

# Export the main API
__all__ = [
    # Tracking
    "lazy",
    "lens",
    "track",
    "update",
    "disable_validation",
    # Type hints and Core classes
    "Lazy",
    "MaybeLazy",
    # Serialization
    "serialize_fingy",
    "deserialize_fingy",
    "save_fingy",
    "load_fingy",
    "prettify_fingy",
    "prettify_serialized_fingy",
    "transpile_fingy",
    # Exceptions
    "ValidationError",
    "SerializationError",
    "DeserializationError",
]
