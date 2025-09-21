import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
import time

class ModelOptimizer:
    """
    Clase para optimizar hiperparámetros del modelo Random Forest
    """
    
    def __init__(self):
        """Inicializar el optimizador"""
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.cv_results = None
        self.optimization_time = None
        
    def optimize_random_forest(self, X_train, y_train, cv_folds=5, 
                              search_type='grid', n_iter=50, verbose=True):
        """
        Optimizar hiperparámetros de Random Forest
        
        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento
            cv_folds: Número de pliegues para validación cruzada
            search_type: Tipo de búsqueda ('grid' o 'random')
            n_iter: Número de iteraciones para búsqueda aleatoria
            verbose: Mostrar progreso
            
        Returns:
            tuple: (mejor_modelo, mejores_parámetros, scores_cv)
        """
        print(f"🔍 Optimizando Random Forest con búsqueda {search_type}...")
        print(f"   - Pliegues CV: {cv_folds}")
        print(f"   - Muestras de entrenamiento: {len(X_train)}")
        
        start_time = time.time()
        
        # Definir grid de parámetros
        param_grid = self._get_param_grid()
        
        if verbose:
            print(f"   - Parámetros a probar: {param_grid}")
            total_combinations = np.prod([len(v) for v in param_grid.values()])
            print(f"   - Combinaciones totales: {total_combinations}")
        
        # Crear modelo base
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Configurar búsqueda
        if search_type == 'grid':
            search = GridSearchCV(
                estimator=rf_base,
                param_grid=param_grid,
                cv=cv_folds,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1 if verbose else 0,
                return_train_score=True
            )
        else:  # random search
            search = RandomizedSearchCV(
                estimator=rf_base,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv_folds,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1 if verbose else 0,
                random_state=42,
                return_train_score=True
            )
        
        # Ejecutar búsqueda
        search.fit(X_train, y_train)
        
        # Guardar resultados
        self.best_model = search.best_estimator_
        self.best_params = search.best_params_
        self.best_score = search.best_score_
        self.cv_results = search.cv_results_
        self.optimization_time = time.time() - start_time
        
        # Validación cruzada adicional con el mejor modelo
        cv_scores = cross_val_score(self.best_model, X_train, y_train, 
                                   cv=cv_folds, scoring='accuracy')
        
        if verbose:
            print(f"Optimización completada en {self.optimization_time:.1f}s")
            print(f"   - Mejor score CV: {self.best_score:.4f}")
            print(f"   - Mejores parámetros: {self.best_params}")
            print(f"   - Scores individuales: {[f'{s:.4f}' for s in cv_scores]}")
            print(f"   - Promedio ± desviación: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return self.best_model, self.best_params, cv_scores
    
    def _get_param_grid(self):
        """
        Definir grid de parámetros para optimización
        
        Returns:
            dict: Grid de parámetros
        """
        # Grid balanceado entre exhaustividad y tiempo de cómputo
        param_grid = {
            'n_estimators': [50, 100, 200],           # Número de árboles
            'max_depth': [10, 15, 20, None],          # Profundidad máxima
            'min_samples_split': [2, 5, 10],          # Mínimo para dividir nodo
            'min_samples_leaf': [1, 2, 4],            # Mínimo en hojas
            'max_features': ['sqrt', 'log2'],         # Features por árbol
            'bootstrap': [True]                        # Muestreo con reemplazo
        }
        
        return param_grid
    
    def optimize_with_custom_scoring(self, X_train, y_train, 
                                    scoring_metric='f1_macro', cv_folds=5):
        """
        Optimizar con métrica personalizada
        
        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento  
            scoring_metric: Métrica a optimizar
            cv_folds: Número de pliegues CV
            
        Returns:
            tuple: (mejor_modelo, mejores_parámetros, scores_cv)
        """
        print(f"Optimizando con métrica: {scoring_metric}")
        
        # Crear scorer personalizado
        if scoring_metric == 'f1_macro':
            scorer = make_scorer(f1_score, average='macro')
        elif scoring_metric == 'f1_weighted':
            scorer = make_scorer(f1_score, average='weighted')
        else:
            scorer = scoring_metric
        
        # Grid más pequeño para métricas complejas
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        search = GridSearchCV(
            estimator=rf_base,
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scorer,
            n_jobs=-1,
            verbose=1
        )
        
        search.fit(X_train, y_train)
        
        self.best_model = search.best_estimator_
        self.best_params = search.best_params_
        self.best_score = search.best_score_
        
        # Validación cruzada con múltiples métricas
        cv_scores = cross_val_score(self.best_model, X_train, y_train, 
                                   cv=cv_folds, scoring=scorer)
        
        print(f"Optimización con {scoring_metric} completada")
        print(f"   - Mejor score: {self.best_score:.4f}")
        print(f"   - Parámetros: {self.best_params}")
        
        return self.best_model, self.best_params, cv_scores
    
    def compare_multiple_configurations(self, X_train, y_train, cv_folds=5):
        """
        Comparar múltiples configuraciones predefinidas
        
        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento
            cv_folds: Número de pliegues CV
            
        Returns:
            dict: Resultados de cada configuración
        """
        print("Comparando múltiples configuraciones predefinidas...")
        
        configurations = {
            'Conservative': {
                'n_estimators': 50,
                'max_depth': 10,
                'min_samples_split': 10,
                'min_samples_leaf': 4
            },
            'Balanced': {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            },
            'Aggressive': {
                'n_estimators': 200,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }
        }
        
        results = {}
        
        for config_name, params in configurations.items():
            print(f"\n   Probando configuración '{config_name}':")
            print(f"   Parámetros: {params}")
            
            # Crear y entrenar modelo
            rf = RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                **params
            )
            
            # Validación cruzada
            cv_scores = cross_val_score(rf, X_train, y_train, 
                                       cv=cv_folds, scoring='accuracy')
            
            results[config_name] = {
                'params': params,
                'cv_scores': cv_scores,
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'model': rf.fit(X_train, y_train)
            }
            
            print(f"   Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Encontrar mejor configuración
        best_config = max(results.keys(), 
                         key=lambda k: results[k]['mean_score'])
        
        print(f"\nMejor configuración: '{best_config}'")
        print(f"   Score: {results[best_config]['mean_score']:.4f}")
        
        self.best_model = results[best_config]['model']
        self.best_params = results[best_config]['params']
        self.best_score = results[best_config]['mean_score']
        
        return results
    
    def analyze_hyperparameter_impact(self, X_train, y_train, param_name, 
                                     param_values, cv_folds=5):
        """
        Analizar el impacto de un hiperparámetro específico
        
        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento
            param_name: Nombre del parámetro a analizar
            param_values: Lista de valores a probar
            cv_folds: Número de pliegues CV
            
        Returns:
            dict: Resultados del análisis
        """
        print(f"📊 Analizando impacto de '{param_name}'")
        print(f"   Valores a probar: {param_values}")
        
        # Parámetros base
        base_params = {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        results = {}
        
        for value in param_values:
            # Actualizar parámetro específico
            current_params = base_params.copy()
            current_params[param_name] = value
            
            # Crear modelo
            rf = RandomForestClassifier(**current_params)
            
            # Validación cruzada
            cv_scores = cross_val_score(rf, X_train, y_train, 
                                       cv=cv_folds, scoring='accuracy')
            
            results[value] = {
                'cv_scores': cv_scores,
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std()
            }
            
            print(f"   {param_name}={value}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Encontrar mejor valor
        best_value = max(results.keys(), 
                        key=lambda k: results[k]['mean_score'])
        
        print(f"Mejor valor para '{param_name}': {best_value}")
        print(f"   Score: {results[best_value]['mean_score']:.4f}")
        
        return results
    
    def get_optimization_summary(self):
        """
        Obtener resumen de la optimización realizada
        
        Returns:
            dict: Resumen de la optimización
        """
        if self.best_model is None:
            return {"status": "not_optimized"}
        
        summary = {
            "status": "optimized",
            "best_score": self.best_score,
            "best_params": self.best_params,
            "optimization_time": self.optimization_time,
            "model_info": {
                "n_estimators": self.best_model.n_estimators,
                "max_depth": self.best_model.max_depth,
                "min_samples_split": self.best_model.min_samples_split,
                "min_samples_leaf": self.best_model.min_samples_leaf
            }
        }
        
        return summary
    
    def save_optimization_results(self, filepath):
        """
        Guardar resultados de optimización
        
        Args:
            filepath: Ruta donde guardar los resultados
        """
        if self.cv_results is None:
            print("No hay resultados de optimización para guardar")
            return
        
        try:
            import pandas as pd
            
            # Convertir resultados a DataFrame
            results_df = pd.DataFrame(self.cv_results)
            
            # Guardar como CSV
            results_df.to_csv(filepath, index=False)
            
            # También guardar resumen en JSON
            summary = self.get_optimization_summary()
            summary_filepath = filepath.replace('.csv', '_summary.json')
            
            import json
            with open(summary_filepath, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"💾 Resultados de optimización guardados:")
            print(f"   - Detalles: {filepath}")
            print(f"   - Resumen: {summary_filepath}")
            
        except Exception as e:
            print(f"Error guardando resultados: {str(e)}")