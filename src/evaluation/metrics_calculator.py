
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from collections import Counter
import json
import os

class MetricsCalculator:
    """
    Clase para calcular métricas de evaluación del modelo
    """
    
    def __init__(self):
        """Inicializar calculador de métricas"""
        self.metrics_history = []
        
    def evaluate_internal_test(self, model, X_test, y_test):
        """
        Evaluar modelo en conjunto de prueba interno (25% del dataset original)
        
        Args:
            model: Modelo entrenado
            X_test: Datos de prueba
            y_test: Etiquetas reales de prueba
            
        Returns:
            dict: Métricas calculadas
        """
        print("Evaluando en conjunto de prueba interno...")
        
        # Predicciones
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Métricas principales
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # Métricas ponderadas
        precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        
        # Reporte de clasificación detallado
        target_names = [f'Dígito {i}' for i in range(10)]
        class_report = classification_report(
            y_test, y_pred, 
            target_names=target_names, 
            output_dict=True,
            zero_division=0
        )
        
        # Calcular métricas por clase
        class_metrics = {}
        for i in range(10):
            if f'Dígito {i}' in class_report:
                class_metrics[i] = class_report[f'Dígito {i}']
        
        metrics = {
            'test_accuracy': accuracy,
            'test_precision_macro': precision_macro,
            'test_recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'test_precision_weighted': precision_weighted,
            'test_recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'class_metrics': class_metrics,
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_proba.tolist(),
            'n_samples': len(y_test)
        }
        
        print(f"Evaluación interna completada:")
        print(f"   - Precisión: {accuracy:.4f}")
        print(f"   - F1-Score macro: {f1_macro:.4f}")
        print(f"   - F1-Score weighted: {f1_weighted:.4f}")
        print(f"   - Muestras evaluadas: {len(y_test)}")
        
        # Guardar en historial
        self.metrics_history.append({
            'type': 'internal_test',
            'metrics': metrics,
            'timestamp': pd.Timestamp.now()
        })
        
        return metrics
    
    def evaluate_custom_images(self, model, custom_images, custom_labels):
        """
        Evaluar modelo en imágenes personalizadas del usuario
        
        Args:
            model: Modelo entrenado
            custom_images: Imágenes procesadas del usuario
            custom_labels: Etiquetas reales de las imágenes
            
        Returns:
            dict: Métricas calculadas para imágenes personalizadas
        """
        if len(custom_images) == 0:
            print("No hay imágenes personalizadas para evaluar")
            return {
                'custom_accuracy': 0,
                'custom_f1_macro': 0,
                'custom_predictions': [],
                'custom_probabilities': [],
                'n_custom_samples': 0
            }
        
        print(f"📸 Evaluando {len(custom_images)} imágenes personalizadas...")
        
        # Predicciones
        y_pred = model.predict(custom_images)
        y_pred_proba = model.predict_proba(custom_images)
        
        # Métricas básicas
        accuracy = accuracy_score(custom_labels, y_pred)
        
        # Métricas adicionales si hay suficientes muestras
        if len(custom_images) >= 3:
            f1_macro = f1_score(custom_labels, y_pred, average='macro', zero_division=0)
            precision_macro = precision_score(custom_labels, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(custom_labels, y_pred, average='macro', zero_division=0)
            
            # Matriz de confusión
            cm = confusion_matrix(custom_labels, y_pred, labels=range(10))
        else:
            f1_macro = 0
            precision_macro = 0
            recall_macro = 0
            cm = np.zeros((10, 10))
        
        # Análisis detallado por imagen
        detailed_results = []
        for i, (true_label, pred_label) in enumerate(zip(custom_labels, y_pred)):
            confidence = np.max(y_pred_proba[i])
            is_correct = true_label == pred_label
            
            detailed_results.append({
                'image_index': i,
                'true_label': int(true_label),
                'predicted_label': int(pred_label),
                'confidence': float(confidence),
                'is_correct': bool(is_correct),
                'probabilities': y_pred_proba[i].tolist()
            })
        
        metrics = {
            'custom_accuracy': accuracy,
            'custom_f1_macro': f1_macro,
            'custom_precision_macro': precision_macro,
            'custom_recall_macro': recall_macro,
            'custom_confusion_matrix': cm.tolist(),
            'custom_predictions': y_pred.tolist(),
            'custom_probabilities': y_pred_proba.tolist(),
            'detailed_results': detailed_results,
            'n_custom_samples': len(custom_images)
        }
        
        print(f"Evaluación de imágenes personalizadas completada:")
        print(f"   - Precisión: {accuracy:.4f}")
        print(f"   - Imágenes correctas: {np.sum(custom_labels == y_pred)}/{len(custom_images)}")
        
        # Mostrar detalles por imagen
        print("   📋 Resultados detallados:")
        for i, result in enumerate(detailed_results):
            status = "✅" if result['is_correct'] else "❌"
            print(f"     Imagen {i+1}: Real={result['true_label']}, "
                  f"Pred={result['predicted_label']}, "
                  f"Conf={result['confidence']:.3f} {status}")
        
        # Guardar en historial
        self.metrics_history.append({
            'type': 'custom_images',
            'metrics': metrics,
            'timestamp': pd.Timestamp.now()
        })
        
        return metrics
    
    def analyze_errors(self, model, X_test, y_test, custom_images=None, custom_labels=None):
        """
        Analizar errores más frecuentes del modelo
        
        Args:
            model: Modelo entrenado
            X_test: Datos de prueba interna
            y_test: Etiquetas de prueba interna
            custom_images: Imágenes personalizadas (opcional)
            custom_labels: Etiquetas personalizadas (opcional)
            
        Returns:
            dict: Análisis de errores
        """
        print("🔍 Analizando errores del modelo...")
        
        error_analysis = {}
        
        # ---- Análisis de errores en conjunto de prueba interno ----
        y_pred_test = model.predict(X_test)
        test_errors = y_test != y_pred_test
        n_test_errors = np.sum(test_errors)
        
        print(f"\n❌ Errores en conjunto de prueba interno:")
        print(f"   - Total de errores: {n_test_errors} de {len(y_test)} ({n_test_errors/len(y_test)*100:.1f}%)")
        
        test_error_analysis = {}
        if n_test_errors > 0:
            # Confusiones más frecuentes
            error_pairs = list(zip(y_test[test_errors], y_pred_test[test_errors]))
            error_counter = Counter(error_pairs)
            
            print("   🔄 Confusiones más frecuentes:")
            test_confusions = []
            for (true_digit, pred_digit), count in error_counter.most_common(10):
                percentage = count / n_test_errors * 100
                test_confusions.append({
                    'true_digit': int(true_digit),
                    'predicted_digit': int(pred_digit),
                    'count': int(count),
                    'percentage': float(percentage)
                })
                print(f"     {true_digit} → {pred_digit}: {count} veces ({percentage:.1f}%)")
            
            test_error_analysis = {
                'total_errors': int(n_test_errors),
                'error_rate': float(n_test_errors / len(y_test)),
                'most_common_confusions': test_confusions
            }
        
        error_analysis['internal_test'] = test_error_analysis
        
        # ---- Análisis de errores en imágenes personalizadas ----
        custom_error_analysis = {}
        if custom_images is not None and len(custom_images) > 0:
            y_pred_custom = model.predict(custom_images)
            custom_errors = custom_labels != y_pred_custom
            n_custom_errors = np.sum(custom_errors)
            
            print(f"\n❌ Errores en imágenes personalizadas:")
            print(f"   - Total de errores: {n_custom_errors} de {len(custom_images)} ({n_custom_errors/len(custom_images)*100:.1f}%)")
            
            if n_custom_errors > 0:
                print("   🔄 Detalles de errores:")
                custom_confusions = []
                for i, (true_label, pred_label) in enumerate(zip(custom_labels[custom_errors], 
                                                                 y_pred_custom[custom_errors])):
                    custom_confusions.append({
                        'image_index': int(i),
                        'true_digit': int(true_label),
                        'predicted_digit': int(pred_label)
                    })
                    print(f"     Imagen {i+1}: {true_label} → {pred_label}")
                
                custom_error_analysis = {
                    'total_errors': int(n_custom_errors),
                    'error_rate': float(n_custom_errors / len(custom_images)),
                    'error_details': custom_confusions
                }
            else:
                custom_error_analysis = {
                    'total_errors': 0,
                    'error_rate': 0.0,
                    'error_details': []
                }
        
        error_analysis['custom_images'] = custom_error_analysis
        
        # ---- Análisis de dígitos más problemáticos ----
        all_errors = []
        if 'most_common_confusions' in test_error_analysis:
            all_errors.extend([(c['true_digit'], c['count']) for c in test_error_analysis['most_common_confusions']])
        
        if custom_error_analysis and 'error_details' in custom_error_analysis:
            for error in custom_error_analysis['error_details']:
                all_errors.append((error['true_digit'], 1))
        
        # Contar errores por dígito
        if all_errors:
            digit_error_counter = Counter([digit for digit, count in all_errors])
            most_problematic = digit_error_counter.most_common(5)
            
            print(f"\n🎯 Dígitos más problemáticos:")
            problematic_digits = []
            for digit, error_count in most_problematic:
                problematic_digits.append({
                    'digit': int(digit),
                    'error_count': int(error_count)
                })
                print(f"     Dígito {digit}: {error_count} errores")
            
            error_analysis['most_problematic_digits'] = problematic_digits
        
        print("✅ Análisis de errores completado")
        
        return error_analysis
    
    def calculate_confidence_statistics(self, model, X_test, custom_images=None):
        """
        Calcular estadísticas de confianza de las predicciones
        
        Args:
            model: Modelo entrenado
            X_test: Datos de prueba
            custom_images: Imágenes personalizadas (opcional)
            
        Returns:
            dict: Estadísticas de confianza
        """
        print("📊 Calculando estadísticas de confianza...")
        
        confidence_stats = {}
        
        # Confianza en conjunto de prueba interno
        y_proba_test = model.predict_proba(X_test)
        confidence_test = np.max(y_proba_test, axis=1)
        
        confidence_stats['internal_test'] = {
            'mean_confidence': float(np.mean(confidence_test)),
            'std_confidence': float(np.std(confidence_test)),
            'min_confidence': float(np.min(confidence_test)),
            'max_confidence': float(np.max(confidence_test)),
            'median_confidence': float(np.median(confidence_test)),
            'low_confidence_count': int(np.sum(confidence_test < 0.5)),
            'high_confidence_count': int(np.sum(confidence_test > 0.8))
        }
        
        # Confianza en imágenes personalizadas
        if custom_images is not None and len(custom_images) > 0:
            y_proba_custom = model.predict_proba(custom_images)
            confidence_custom = np.max(y_proba_custom, axis=1)
            
            confidence_stats['custom_images'] = {
                'mean_confidence': float(np.mean(confidence_custom)),
                'std_confidence': float(np.std(confidence_custom)),
                'min_confidence': float(np.min(confidence_custom)),
                'max_confidence': float(np.max(confidence_custom)),
                'median_confidence': float(np.median(confidence_custom)),
                'low_confidence_count': int(np.sum(confidence_custom < 0.5)),
                'high_confidence_count': int(np.sum(confidence_custom > 0.8))
            }
        
        print("✅ Estadísticas de confianza calculadas")
        
        return confidence_stats
    
    def save_metrics(self, metrics_dict, filepath):
        """
        Guardar métricas en archivo JSON
        
        Args:
            metrics_dict: Diccionario con métricas
            filepath: Ruta donde guardar
        """
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Convertir arrays numpy a listas para serialización JSON
            serializable_metrics = self._make_json_serializable(metrics_dict.copy())
            
            # Agregar timestamp
            serializable_metrics['timestamp'] = pd.Timestamp.now().isoformat()
            
            # Guardar
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Métricas guardadas en: {filepath}")
            
        except Exception as e:
            print(f"❌ Error guardando métricas: {str(e)}")
    
    def _make_json_serializable(self, obj):
        """
        Convertir objeto a formato serializable en JSON
        
        Args:
            obj: Objeto a convertir
            
        Returns:
            Objeto serializable
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def generate_metrics_report(self, metrics_dict):
        """
        Generar reporte textual de métricas
        
        Args:
            metrics_dict: Diccionario con métricas
            
        Returns:
            str: Reporte formateado
        """
        report_lines = []
        report_lines.append("📊 REPORTE DE MÉTRICAS DE EVALUACIÓN")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Métricas del conjunto de prueba interno
        if 'test_accuracy' in metrics_dict:
            report_lines.append("🔹 CONJUNTO DE PRUEBA INTERNO:")
            report_lines.append(f"   - Precisión: {metrics_dict['test_accuracy']:.4f}")
            report_lines.append(f"   - F1-Score macro: {metrics_dict.get('f1_macro', 0):.4f}")
            report_lines.append(f"   - F1-Score weighted: {metrics_dict.get('f1_weighted', 0):.4f}")
            report_lines.append(f"   - Muestras evaluadas: {metrics_dict.get('n_samples', 'N/A')}")
            report_lines.append("")
        
        # Métricas de imágenes personalizadas
        if 'custom_accuracy' in metrics_dict:
            report_lines.append("🔹 IMÁGENES PERSONALIZADAS:")
            report_lines.append(f"   - Precisión: {metrics_dict['custom_accuracy']:.4f}")
            report_lines.append(f"   - F1-Score macro: {metrics_dict.get('custom_f1_macro', 0):.4f}")
            report_lines.append(f"   - Muestras evaluadas: {metrics_dict.get('n_custom_samples', 'N/A')}")
            report_lines.append("")
        
        # Análisis de errores si está disponible
        if 'error_analysis' in metrics_dict:
            error_analysis = metrics_dict['error_analysis']
            report_lines.append("🔹 ANÁLISIS DE ERRORES:")
            
            if 'internal_test' in error_analysis:
                internal_errors = error_analysis['internal_test']
                if 'total_errors' in internal_errors:
                    report_lines.append(f"   - Errores prueba interna: {internal_errors['total_errors']}")
                    report_lines.append(f"   - Tasa de error: {internal_errors.get('error_rate', 0)*100:.1f}%")
            
            if 'custom_images' in error_analysis:
                custom_errors = error_analysis['custom_images']
                if 'total_errors' in custom_errors:
                    report_lines.append(f"   - Errores imágenes propias: {custom_errors['total_errors']}")
                    report_lines.append(f"   - Tasa de error: {custom_errors.get('error_rate', 0)*100:.1f}%")
            
            report_lines.append("")
        
        return "\n".join(report_lines)