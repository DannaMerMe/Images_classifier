"""
Módulo de procesamiento de datos
==============================

Contiene clases para:
- Carga del dataset load_digits
- Procesamiento de imágenes personalizadas
- Transformaciones y normalización
"""

from .data_loader import DataLoader
from .image_processor import ImageProcessor

__all__ = ['DataLoader', 'ImageProcessor']