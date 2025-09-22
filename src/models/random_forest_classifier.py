import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
from sklearn.metrics import accuracy_score, classification_report

class RandomForestClassifier:
    """
    Wrapper para Random Forest con funcionalidades específicas del proyecto
    """
    
    def __init__(self, **kwargs):
        """
        Inicializar el clasificador
        
        Args:
            **kwargs: Parámetros para RandomForestClassifier de sklearn
        """
        # Parámetros por defecto justificados técnicamente
        default_params = {
            'n_estimators': 100,        # Balance entre precisión y tiempo de cómputo
            'max_depth': 15,            # Evita overfitting manteniendo expresividad
            'min_samples_split': 5,     # Reduce ruido en divisiones
            'min_samples_leaf': 2,      # Mejora generalización
            'random_state': 42,         # Reproducibilidad
            'n_jobs': -1,              # Paralelización máxima
            'class_weight': 'balanced'  # Manejo de clases desbalanceadas
        }
        
        # Combinar parámetros por defecto con los proporcionados
        self.params = {**default_params, **kwargs}
        self.model = None
        self.is_trained = False
        self.feature_importances = None
        
        print("🌲 Random Forest Classifier inicializado")
        print(f"   Parámetros: {self.params}")
        
    def train_base_model(self, X_train, y_train, X_test=None, y_test=None):
        """
        Entrenar modelo con parámetros base
        
        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento  
            X_test: Datos de prueba (opcional)
            y_test: Etiquetas de prueba (opcional)
            
        Returns:
            float: Precisión en conjunto de prueba (si se proporciona)
        """
        print("🔧 Entrenando Random Forest con parámetros base...")
        
        # Crear y entrenar modelo
        self.model = SklearnRandomForest(**self.params)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Guardar importancias de características
        self.feature_importances = self.model.feature_importances_
        
        # Evaluación en entrenamiento
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        print(f"Modelo entrenado exitosamente")
        print(f"   - Precisión en entrenamiento: {train_accuracy:.4f}")
        print(f"   - Número de árboles: {self.model.n_estimators}")
        print(f"   - Características consideradas: {X_train.shape[1]}")
        
        # Evaluación en prueba si se proporciona
        test_accuracy = None
        if X_test is not None and y_test is not None:
            test_pred = self.model.predict(X_test)
            test_accuracy = accuracy_score(y_test, test_pred)
            print(f"   - Precisión en prueba: {test_accuracy:.4f}")
            
            # Verificar overfitting
            overfitting_gap = train_accuracy - test_accuracy
            if overfitting_gap > 0.1:
                print(f"Posible overfitting detectado (gap: {overfitting_gap:.4f})")
            else:
                print("No hay signos significativos de overfitting")
        
        return test_accuracy
    
    def predict(self, X):
        """
        Realizar predicciones
        
        Args:
            X: Datos de entrada
            
        Returns:
            np.array: Predicciones
        """
        if not self.is_trained:
            raise ValueError("Modelo no entrenado. Ejecuta train_base_model() primero.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predecir probabilidades de clase
        
        Args:
            X: Datos de entrada
            
        Returns:
            np.array: Probabilidades por clase
        """
        if not self.is_trained:
            raise ValueError("Modelo no entrenado. Ejecuta train_base_model() primero.")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, top_n=10):
        """
        Obtener importancia de características más relevantes
        
        Args:
            top_n: Número de características más importantes a mostrar
            
        Returns:
            dict: Diccionario con índices y valores de importancia
        """
        if self.feature_importances is None:
            print("Modelo no entrenado o sin importancias calculadas")
            return {}
        
        # Obtener índices ordenados por importancia (descendente)
        sorted_indices = np.argsort(self.feature_importances)[::-1]
        
        print(f"Top {top_n} características más importantes:")
        
        importance_dict = {}
        for i in range(min(top_n, len(sorted_indices))):
            idx = sorted_indices[i]
            importance = self.feature_importances[idx]
            importance_dict[idx] = importance
            
            # Convertir índice a posición en imagen 8x8
            row, col = idx // 8, idx % 8
            print(f"   {i+1}. Píxel [{row},{col}] (índice {idx}): {importance:.4f}")
        
        return importance_dict
    
    def evaluate_detailed(self, X_test, y_test, target_names=None):
        """
        Evaluación detallada del modelo
        
        Args:
            X_test: Datos de prueba
            y_test: Etiquetas reales
            target_names: Nombres de las clases (opcional)
            
        Returns:
            dict: Métricas detalladas
        """
        if not self.is_trained:
            raise ValueError("Modelo no entrenado.")
        
        # Predicciones
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        # Métricas básicas
        accuracy = accuracy_score(y_test, y_pred)
        
        # Reporte de clasificación
        if target_names is None:
            target_names = [f'Dígito {i}' for i in range(10)]
        
        report = classification_report(y_test, y_pred, 
                                     target_names=target_names, 
                                     output_dict=True)
        
        print("Evaluación detallada:")
        print(f"   - Precisión general: {accuracy:.4f}")
        print(f"   - Macro avg F1-score: {report['macro avg']['f1-score']:.4f}")
        print(f"   - Weighted avg F1-score: {report['weighted avg']['f1-score']:.4f}")
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': report
        }
    
    def save_model(self, filepath):
        """
        Guardar modelo entrenado
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        if not self.is_trained:
            print("Modelo no entrenado. No se puede guardar.")
            return
        
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Guardar modelo usando joblib (más eficiente para sklearn)
            joblib.dump(self.model, filepath)
            
            # También guardar parámetros y metadatos
            metadata = {
                'params': self.params,
                'feature_importances': self.feature_importances.tolist() if self.feature_importances is not None else None,
                'n_features': self.model.n_features_in_,
                'n_classes': len(self.model.classes_)
            }
            
            metadata_filepath = filepath.replace('.pkl', '_metadata.json')
            import json
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Modelo guardado exitosamente:")
            print(f"   - Modelo: {filepath}")
            print(f"   - Metadatos: {metadata_filepath}")
            
        except Exception as e:
            print(f"Error guardando modelo: {str(e)}")
    
    def load_model(self, filepath):
        """
        Cargar modelo previamente guardado
        
        Args:
            filepath: Ruta del modelo a cargar
        """
        try:
            # Cargar modelo
            self.model = joblib.load(filepath)
            self.is_trained = True
            
            # Intentar cargar metadatos
            metadata_filepath = filepath.replace('.pkl', '_metadata.json')
            if os.path.exists(metadata_filepath):
                import json
                with open(metadata_filepath, 'r') as f:
                    metadata = json.load(f)
                
                self.params = metadata.get('params', {})
                importances = metadata.get('feature_importances')
                if importances:
                    self.feature_importances = np.array(importances)
                
                print(f" Modelo y metadatos cargados desde: {filepath}")
            else:
                print(f"Modelo cargado desde: {filepath} (sin metadatos)")
            
            print(f"Modelo listo para usar")
            print(f"   - Características: {self.model.n_features_in_}")
            print(f"   - Clases: {len(self.model.classes_)}")
            
        except Exception as e:
            print(f"Error cargando modelo: {str(e)}")
            self.model = None
            self.is_trained = False
    
    def get_model_info(self):
        """
        Obtener información del modelo actual
        
        Returns:
            dict: Información del modelo
        """
        if not self.is_trained:
            return {"status": "not_trained"}
        
        info = {
            "status": "trained",
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "n_features": self.model.n_features_in_,
            "n_classes": len(self.model.classes_),
            "classes": self.model.classes_.tolist(),
            "parameters": self.params
        }
        
        return info