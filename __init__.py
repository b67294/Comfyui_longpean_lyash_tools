"""Top-level package for visual_perspectivemixer."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """Lyash"""
__email__ = "b67294@shu.edu.cn"
__version__ = "0.0.1"

from .src.visual_perspectivemixer.nodes import NODE_CLASS_MAPPINGS as BASE_NODE_CLASS_MAPPINGS
from .src.visual_perspectivemixer.nodes import NODE_DISPLAY_NAME_MAPPINGS as BASE_NODE_DISPLAY_NAME_MAPPINGS
from .src.visual_perspectivemixer.nodes_advanced import NODE_CLASS_MAPPINGS as ADV_NODE_CLASS_MAPPINGS
from .src.visual_perspectivemixer.nodes_advanced import NODE_DISPLAY_NAME_MAPPINGS as ADV_NODE_DISPLAY_NAME_MAPPINGS
from .src.tissue_paper_alpha_modifier import NODE_CLASS_MAPPINGS as TISSUE_NODE_CLASS_MAPPINGS
from .src.tissue_paper_alpha_modifier import NODE_DISPLAY_NAME_MAPPINGS as TISSUE_NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {
    **BASE_NODE_CLASS_MAPPINGS,
    **ADV_NODE_CLASS_MAPPINGS,
    **TISSUE_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **BASE_NODE_DISPLAY_NAME_MAPPINGS,
    **ADV_NODE_DISPLAY_NAME_MAPPINGS,
    **TISSUE_NODE_DISPLAY_NAME_MAPPINGS,
}

WEB_DIRECTORY = "./web"
