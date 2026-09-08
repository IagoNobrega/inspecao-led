"""Núcleo do sistema de inspeção visual de LEDs."""

from .core import (
    InspectionResult,
    LedRegion,
    align_image,
    annotate_image,
    generate_demo_images,
    generate_grid_regions,
    inspect_leds,
    inspect_leds_buffer,
)
from .logging_config import get_logger, setup_logging
from .ollama import (
    DEFAULT_OLLAMA_URL,
    DEFAULT_VISION_MODEL,
    LlmInspection,
    OllamaError,
    analyze_with_vision,
    list_vision_models,
)

__all__ = [
    "InspectionResult",
    "LedRegion",
    "align_image",
    "annotate_image",
    "generate_demo_images",
    "generate_grid_regions",
    "inspect_leds",
    "inspect_leds_buffer",
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_VISION_MODEL",
    "LlmInspection",
    "OllamaError",
    "analyze_with_vision",
    "list_vision_models",
    "setup_logging",
    "get_logger",
]
