import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataLoader:
    """
    Clase para cargar y preparar el dataset load_digits
    """
    
    def __init__(self):
        """Inicializar el cargador de datos"""
        self.digits_data = None
        self.scaler = StandardScaler()
        
    def load_digits_dataset(self, test_size=0.25, random_state=42, normalize=False):
        """
        Cargar el dataset load_digits y dividirlo en entrenamiento/prueba
        
        Args:
            test_size: Proporción del conjunto de prueba (default: 0.25)
            random_state: Semilla para reproducibilidad (default: 42)
            normalize: Si normalizar los datos (default: False)
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print("Cargando dataset load_digits...")
        
        # Cargar dataset
        self.digits_data = load_digits()
        
        print(f"   - Total de muestras: {self.digits_data.data.shape[0]}")
        print(f"   - Dimensiones por imagen: {self.digits_data.data.shape[1]} (8x8 píxeles)")
        print(f"   - Número de clases: {len(np.unique(self.digits_data.target))}")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            self.digits_data.data,
            self.digits_data.target,
            test_size=test_size,
            random_state=random_state,
            stratify=self.digits_data.target  # Mantener distribución de clases
        )
        
        # Normalizar si se solicita
        if normalize:
            print("   - Normalizando datos...")
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        print(f"División completada:")
        print(f"   - Entrenamiento: {X_train.shape[0]} muestras")
        print(f"   - Prueba: {X_test.shape[0]} muestras")
        
        return X_train, X_test, y_train, y_test
    
    def get_sample_images(self, n_samples=10):
        """
        Obtener muestras del dataset para visualización
        
        Args:
            n_samples: Número de muestras a obtener
            
        Returns:
            tuple: images, labels
        """
        if self.digits_data is None:
            raise ValueError("Dataset no cargado. Ejecuta load_digits_dataset() primero.")
        
        indices = np.random.choice(len(self.digits_data.data), n_samples, replace=False)
        return self.digits_data.data[indices], self.digits_data.target[indices]
    
    def get_class_distribution(self):
        """
        Obtener distribución de clases en el dataset
        
        Returns:
            dict: Distribución de clases
        """
        if self.digits_data is None:
            raise ValueError("Dataset no cargado. Ejecuta load_digits_dataset() primero.")
        
        unique, counts = np.unique(self.digits_data.target, return_counts=True)
        distribution = dict(zip(unique, counts))
        
        print("Distribución de clases:")
        for digit, count in distribution.items():
            print(f"   Dígito {digit}: {count} muestras")
        
        return distribution