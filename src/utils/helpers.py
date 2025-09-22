import os
import numpy as np
from datetime import datetime

def create_directories(dir_list):
    """
    Crear directorios si no existen
    
    Args:
        dir_list: Lista de rutas de directorios a crear
    """
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directorio creado: {directory}")
        else:
            print(f"Directorio ya existe: {directory}")

def print_banner(title):
    """
    Imprimir banner decorativo
    
    Args:
        title: Título a mostrar
    """
    width = max(60, len(title) + 10)
    print("=" * width)
    print(f"{title:^{width}}")
    print("=" * width)

def print_step(step_num, description):
    """
    Imprimir paso del pipeline con formato
    
    Args:
        step_num: Número del paso
        description: Descripción del paso
    """
    print(f"\nPASO {step_num}: {description}")
    print("-" * (len(description) + 15))

def format_time(seconds):
    """
    Formatear tiempo en formato legible
    
    Args:
        seconds: Tiempo en segundos
        
    Returns:
        Tiempo formateado como string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"

def save_array_to_file(array, filename, directory):
    """
    Guardar array numpy en archivo
    
    Args:
        array: Array numpy a guardar
        filename: Nombre del archivo
        directory: Directorio donde guardar
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filepath = os.path.join(directory, filename)
    np.save(filepath, array)
    print(f"Array guardado: {filepath}")

def load_array_from_file(filepath):
    """
    Cargar array numpy desde archivo
    
    Args:
        filepath: Ruta del archivo
        
    Returns:
        Array numpy cargado
    """
    if os.path.exists(filepath):
        array = np.load(filepath)
        print(f"Array cargado: {filepath}")
        return array
    else:
        print(f"Archivo no encontrado: {filepath}")
        return None

def get_timestamp():
    """
    Obtener timestamp actual
    
    Returns:
        Timestamp como string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def extract_digit_from_filename(filename):
    """
    Extraer dígito del nombre de archivo
    
    Args:
        filename: Nombre del archivo
        
    Returns:
        Dígito extraído o None si no se encuentra
    """
    for char in filename:
        if char.isdigit():
            return int(char)
    return None

def validate_image_dimensions(image, expected_shape=(8, 8)):
    """
    Validar dimensiones de imagen
    
    Args:
        image: Imagen como array numpy
        expected_shape: Forma esperada
        
    Returns:
        True si las dimensiones son correctas
    """
    if image.shape == expected_shape:
        return True
    else:
        print(f"Dimensiones incorrectas: {image.shape}, esperado: {expected_shape}")
        return False