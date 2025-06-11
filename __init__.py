"""
ComfyUI Image & Prompt Loader
A powerful custom node for loading images with automatic prompt extraction from multiple sources.

Features:
- Civitai integration with automatic metadata extraction
- Smart caption file detection for datasets  
- EXIF metadata support
- Dynamic preview updates
- Multiple input modes

Author: hassan-sd
Repository: https://github.com/hassan-sd/comfyui-image-prompt-loader
License: Apache-2.0
"""

from .image_prompt_loader import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Required for ComfyUI to recognize this as a valid custom node package
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Version and metadata for ComfyUI Manager
__version__ = "1.0.0"
WEB_DIRECTORY = None  # No web components needed
