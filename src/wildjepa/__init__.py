"""WildJEPA: I-JEPA-based self-supervised species classification for camera-trap imagery."""

import warnings

# This env's torchvision build wants libjpeg.9.dylib (the standard IJG jpeg
# ABI); only libjpeg-turbo (libjpeg.8.dylib -- a different, incompatible
# ABI) is installed, so torchvision.io's native image extension fails to
# dlopen and warns on every import. Purely cosmetic: this project loads
# images via PIL (see data/iwildcam.py), never torchvision.io, and fixing it
# properly would mean installing conda-forge's separate `jpeg` package
# alongside libjpeg-turbo, risking a real, working dependency (Pillow's own
# JPEG support) for a warning with zero functional impact. Filtered here,
# not at each entry point, since every import path (scripts/*.py, tests/)
# passes through this package's __init__ before anything else.
warnings.filterwarnings("ignore", message="Failed to load image Python extension", category=UserWarning)

__version__ = "0.1.0"
