"""Top-level package for visual_perspectivemixer."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """Lyash"""
__email__ = "b67294@shu.edu.cn"
__version__ = "0.0.1"

from .src.visual_perspectivemixer.nodes import NODE_CLASS_MAPPINGS
from .src.visual_perspectivemixer.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
