import numpy as np
import cv2
import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

class ImageProcessor:
    """
    Clase para procesar imágenes personalizadas del usuario
    """
    
    def __init__(self):
        """Inicializar el procesador de imágenes"""
        self.processed_images = []
        self.image_labels = []
        self.image_names = []
        
    def process_custom_images(self, data_folder, image_extensions=None):
        """
        Procesar todas las imágenes personalizadas en una carpeta
        
        Args:
            data_folder: Carpeta que contiene las imágenes
            image_extensions: Lista de extensiones soportadas
            
        Returns:
            tuple: (processed_images, labels, image_names)
        """
        if image_extensions is None:
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        
        print(f"📸 Procesando imágenes personalizadas desde '{data_folder}'...")
        
        # Limpiar listas
        self.processed_images = []
        self.image_labels = []
        self.image_names = []
        
        # Buscar archivos de imagen
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob(os.path.join(data_folder, ext)))
            image_paths.extend(glob(os.path.join(data_folder, '**', ext), recursive=True))
        
        print(f"   - Archivos encontrados: {len(image_paths)}")
        
        if len(image_paths) == 0:
            print("No se encontraron imágenes en la carpeta especificada")
            print(f"Verifica que existan archivos con extensiones: {image_extensions}")
            return np.array([]), np.array([]), []
        
        # Procesar cada imagen
        successful_count = 0
        for img_path in image_paths:
            try:
                # Extraer etiqueta del nombre del archivo
                filename = os.path.basename(img_path)
                label = self._extract_label_from_filename(filename)
                
                if label is None:
                    print(f"No se pudo extraer etiqueta de '{filename}'")
                    continue
                
                # Procesar imagen
                processed_img = self._process_single_image(img_path)
                
                if processed_img is not None:
                    self.processed_images.append(processed_img)
                    self.image_labels.append(label)
                    self.image_names.append(filename)
                    successful_count += 1
                    print(f"{filename} → dígito {label}")
                else:
                    print(f"Error procesando {filename}")
                    
            except Exception as e:
                print(f"Error con {os.path.basename(img_path)}: {str(e)}")
        
        # Convertir a arrays numpy
        if len(self.processed_images) > 0:
            processed_array = np.array(self.processed_images)
            labels_array = np.array(self.image_labels)
            
            print(f"Procesamiento completado:")
            print(f"   - Imágenes procesadas exitosamente: {successful_count}")
            print(f"   - Distribución por dígito:")
            
            for digit in range(10):
                count = np.sum(labels_array == digit)
                if count > 0:
                    print(f"     Dígito {digit}: {count} imagen(es)")
                    
            return processed_array, labels_array, self.image_names
        else:
            print("No se procesaron imágenes exitosamente")
            return np.array([]), np.array([]), []
    
    def _process_single_image(self, image_path):
        """
        Procesar una imagen individual
        
        Args:
            image_path: Ruta de la imagen
            
        Returns:
            Array numpy de la imagen procesada (64 elementos) o None si hay error
        """
        try:
            # Cargar imagen en escala de grises
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # Intentar con PIL si OpenCV falla
                img = Image.open(image_path).convert('L')
                img = np.array(img)
            
            # Redimensionar a 8x8 píxeles (compatible con load_digits)
            img_resized = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
            
            # Normalizar contraste: invertir si es necesario
            # load_digits tiene texto claro sobre fondo oscuro
            if np.mean(img_resized) > 127:  # Fondo claro
                img_resized = 255 - img_resized
            
            # Normalizar a rango 0-16 (como load_digits)
            img_normalized = (img_resized / 255.0 * 16).astype(np.float64)
            
            # Aplanar a vector 1D (64 elementos)
            img_flattened = img_normalized.flatten()
            
            # Verificar dimensiones
            if len(img_flattened) != 64:
                raise ValueError(f"Dimensión incorrecta: {len(img_flattened)}, esperado: 64")
            
            return img_flattened
            
        except Exception as e:
            print(f"Error procesando {image_path}: {str(e)}")
            return None
    
    def _extract_label_from_filename(self, filename):
        """
        Extraer etiqueta numérica del nombre de archivo
        
        Args:
            filename: Nombre del archivo
            
        Returns:
            int: Dígito extraído (0-9) o None si no se encuentra
        """
        # Buscar el primer dígito en el nombre del archivo
        for char in filename:
            if char.isdigit():
                digit = int(char)
                if 0 <= digit <= 9:
                    return digit
        
        return None
    
    def save_processed_images(self, save_dir):
        """
        Guardar imágenes procesadas como arrays numpy
        
        Args:
            save_dir: Directorio donde guardar
        """
        if len(self.processed_images) == 0:
            print("No hay imágenes procesadas para guardar")
            return
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Guardar arrays
        images_array = np.array(self.processed_images)
        labels_array = np.array(self.image_labels)
        
        np.save(os.path.join(save_dir, 'processed_images.npy'), images_array)
        np.save(os.path.join(save_dir, 'processed_labels.npy'), labels_array)
        
        # Guardar nombres de archivo
        with open(os.path.join(save_dir, 'image_names.txt'), 'w') as f:
            for name in self.image_names:
                f.write(f"{name}\n")
        
        print(f"Imágenes procesadas guardadas en: {save_dir}")
    
    def load_processed_images(self, save_dir):
        """
        Cargar imágenes procesadas previamente guardadas
        
        Args:
            save_dir: Directorio desde donde cargar
            
        Returns:
            tuple: (processed_images, labels, image_names)
        """
        try:
            images_path = os.path.join(save_dir, 'processed_images.npy')
            labels_path = os.path.join(save_dir, 'processed_labels.npy')
            names_path = os.path.join(save_dir, 'image_names.txt')
            
            if not all(os.path.exists(path) for path in [images_path, labels_path, names_path]):
                print("Archivos de imágenes procesadas no encontrados")
                return np.array([]), np.array([]), []
            
            # Cargar arrays
            processed_images = np.load(images_path)
            labels = np.load(labels_path)
            
            # Cargar nombres
            with open(names_path, 'r') as f:
                image_names = [line.strip() for line in f.readlines()]
            
            self.processed_images = processed_images.tolist()
            self.image_labels = labels.tolist()
            self.image_names = image_names
            
            print(f"Imágenes procesadas cargadas desde: {save_dir}")
            print(f"   - {len(processed_images)} imágenes cargadas")
            
            return processed_images, labels, image_names
            
        except Exception as e:
            print(f"Error cargando imágenes procesadas: {str(e)}")
            return np.array([]), np.array([]), []
    
    def validate_processed_images(self, images, labels):
        """
        Validar que las imágenes procesadas tienen el formato correcto
        
        Args:
            images: Array de imágenes procesadas
            labels: Array de etiquetas
            
        Returns:
            bool: True si la validación es exitosa
        """
        print("🔍 Validando imágenes procesadas...")
        
        # Verificar que no estén vacías
        if len(images) == 0:
            print("No hay imágenes para validar")
            return False
        
        # Verificar dimensiones
        if images.shape[1] != 64:
            print(f"Dimensión incorrecta: {images.shape[1]}, esperado: 64")
            return False
        
        # Verificar que labels y images tengan la misma cantidad
        if len(images) != len(labels):
            print(f"Desajuste: {len(images)} imágenes vs {len(labels)} etiquetas")
            return False
        
        # Verificar rango de valores
        min_val, max_val = images.min(), images.max()
        if min_val < 0 or max_val > 16:
            print(f"Rango de valores inusual: [{min_val:.2f}, {max_val:.2f}], esperado: [0, 16]")
        
        # Verificar rango de etiquetas
        unique_labels = np.unique(labels)
        invalid_labels = [l for l in unique_labels if l < 0 or l > 9]
        if invalid_labels:
            print(f"Etiquetas inválidas encontradas: {invalid_labels}")
            return False
        
        print("Validación exitosa")
        print(f"   - {len(images)} imágenes válidas")
        print(f"   - Etiquetas únicas: {sorted(unique_labels.tolist())}")
        print(f"   - Rango de valores: [{min_val:.2f}, {max_val:.2f}]")
        
        return True