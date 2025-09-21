#!/usr/bin/env python3
import os
import sys
import warnings
from datetime import datetime

# Suprimir warnings innecesarios
warnings.filterwarnings('ignore')

# Agregar src al path para imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Imports del proyecto
from src.utils.config import Config
from src.utils.helpers import create_directories, print_banner, print_step
from src.data_processing.data_loader import DataLoader
from src.data_processing.image_processor import ImageProcessor
from src.models.random_forest_classifier import RandomForestClassifier
from src.models.model_optimizer import ModelOptimizer
from src.evaluation.metrics_calculator import MetricsCalculator
from src.evaluation.visualization import Visualizer


class DigitClassificationPipeline:
    """
    Pipeline principal para la clasificación de dígitos manuscritos
    """
    
    def __init__(self):
        """Inicializar el pipeline"""
        self.config = Config()
        self.data_loader = DataLoader()
        self.image_processor = ImageProcessor()
        self.classifier = RandomForestClassifier()
        self.optimizer = ModelOptimizer()
        self.metrics = MetricsCalculator()
        self.visualizer = Visualizer()
        
        # Variables para almacenar resultados
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.custom_images = None
        self.custom_labels = None
        self.results = {}
        
    def setup_environment(self):
        """Configurar el ambiente de trabajo"""
        print_step(1, "Configurando ambiente de trabajo")
        
        # Crear directorios necesarios
        create_directories([
            self.config.DATA_RAW_DIR,
            self.config.DATA_PROCESSED_DIR,
            self.config.RESULTS_MODELS_DIR,
            self.config.RESULTS_PLOTS_DIR,
            self.config.RESULTS_METRICS_DIR,
            self.config.RESULTS_REPORTS_DIR
        ])
        
        print("Ambiente configurado correctamente")
        
    def load_and_prepare_data(self):
        """Cargar y preparar los datos"""
        print_step(2, "Cargando y preparando datos")
        
        # Cargar dataset load_digits
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_loader.load_digits_dataset(
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE
        )
        
        # Cargar y procesar imágenes propias
        self.custom_images, self.custom_labels, image_names = self.image_processor.process_custom_images(
            self.config.DATA_RAW_DIR
        )
        
        print(f"Datos cargados:")
        print(f"   - Entrenamiento: {len(self.X_train)} muestras")
        print(f"   - Prueba: {len(self.X_test)} muestras")
        print(f"   - Imágenes propias: {len(self.custom_images)} muestras")
        
        return image_names
        
    def train_base_model(self):
        """Entrenar modelo base"""
        print_step(3, "Entrenando modelo Random Forest base")
        
        # Entrenar modelo con parámetros por defecto
        base_accuracy = self.classifier.train_base_model(self.X_train, self.y_train, self.X_test, self.y_test)
        
        print(f"Modelo base entrenado con precisión: {base_accuracy:.4f}")
        
        return base_accuracy
        
    def optimize_model(self):
        """Optimizar hiperparámetros del modelo"""
        print_step(4, "Optimizando hiperparámetros con validación cruzada")
        
        # Optimizar modelo
        self.best_model, best_params, cv_scores = self.optimizer.optimize_random_forest(
            self.X_train, self.y_train,
            cv_folds=self.config.CV_FOLDS
        )
        
        self.results['best_params'] = best_params
        self.results['cv_scores'] = cv_scores
        
        print(f"Modelo optimizado:")
        print(f"   - Mejores parámetros: {best_params}")
        print(f"   - CV Score promedio: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return best_params, cv_scores
        
    def evaluate_model(self):
        """Evaluar el modelo en ambos conjuntos de datos"""
        print_step(5, "Evaluando modelo")
        
        # Evaluar en conjunto de prueba interno
        internal_metrics = self.metrics.evaluate_internal_test(
            self.best_model, self.X_test, self.y_test
        )
        
        # Evaluar en imágenes propias
        custom_metrics = self.metrics.evaluate_custom_images(
            self.best_model, self.custom_images, self.custom_labels
        )
        
        # Guardar métricas
        self.results.update(internal_metrics)
        self.results.update(custom_metrics)
        
        print(f"Evaluación completada:")
        print(f"   - Precisión prueba interna: {internal_metrics.get('test_accuracy', 0):.4f}")
        print(f"   - Precisión imágenes propias: {custom_metrics.get('custom_accuracy', 0):.4f}")
        
        return internal_metrics, custom_metrics
        
    def analyze_errors(self):
        """Analizar errores del modelo"""
        print_step(6, "Analizando errores")
        
        # Análisis de errores
        error_analysis = self.metrics.analyze_errors(
            self.best_model, 
            self.X_test, self.y_test,
            self.custom_images, self.custom_labels
        )
        
        self.results['error_analysis'] = error_analysis
        
        print("Análisis de errores completado")
        
        return error_analysis
        
    def generate_visualizations(self, image_names):
        """Generar visualizaciones"""
        print_step(7, "Generando visualizaciones")
        
        # Visualizar imágenes procesadas
        if len(self.custom_images) > 0:
            self.visualizer.plot_custom_images(
                self.custom_images, self.custom_labels, image_names,
                save_path=os.path.join(self.config.RESULTS_PLOTS_DIR, 'custom_images.png')
            )
        
        # Matriz de confusión - prueba interna
        y_pred_test = self.best_model.predict(self.X_test)
        self.visualizer.plot_confusion_matrix(
            self.y_test, y_pred_test, 
            title="Matriz de Confusión - Conjunto Prueba Interno",
            save_path=os.path.join(self.config.RESULTS_PLOTS_DIR, 'confusion_matrix_internal.png')
        )
        
        # Matriz de confusión - imágenes propias
        if len(self.custom_images) > 0:
            y_pred_custom = self.best_model.predict(self.custom_images)
            self.visualizer.plot_confusion_matrix(
                self.custom_labels, y_pred_custom,
                title="Matriz de Confusión - Imágenes Propias",
                save_path=os.path.join(self.config.RESULTS_PLOTS_DIR, 'confusion_matrix_custom.png')
            )
        
        # Gráfico de validación cruzada
        self.visualizer.plot_cross_validation_scores(
            self.results['cv_scores'],
            save_path=os.path.join(self.config.RESULTS_PLOTS_DIR, 'cv_scores.png')
        )
        
        print("Visualizaciones generadas y guardadas")
        
    def save_results(self):
        """Guardar resultados y modelo"""
        print_step(8, "Guardando resultados")
        
        # Guardar modelo
        model_path = os.path.join(self.config.RESULTS_MODELS_DIR, 'best_random_forest_model.pkl')
        self.classifier.save_model(self.best_model, model_path)
        
        # Guardar métricas
        metrics_path = os.path.join(self.config.RESULTS_METRICS_DIR, 'evaluation_metrics.json')
        self.metrics.save_metrics(self.results, metrics_path)
        
        print("Resultados guardados")
        
    def generate_final_report(self):
        """Generar informe final"""
        print_step(9, "Generando informe final")
        
        # Crear informe
        report_content = self._create_report_content()
        
        # Guardar informe
        report_path = os.path.join(self.config.RESULTS_REPORTS_DIR, 'informe_final.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Mostrar informe en consola
        print("\n" + "="*80)
        print(report_content)
        print("="*80)
        
        print(f"Informe guardado en: {report_path}")
        
    def _create_report_content(self):
        """Crear contenido del informe final"""
        report = []
        report.append("INFORME FINAL - CLASIFICACIÓN DE DÍGITOS MANUSCRITOS")
        report.append("=" * 60)
        report.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Resumen del modelo
        report.append("MODELO IMPLEMENTADO:")
        report.append(f"   - Algoritmo: Random Forest")
        report.append(f"   - Parámetros optimizados: {self.results.get('best_params', 'N/A')}")
        report.append("")
        
        # Validación cruzada
        cv_scores = self.results.get('cv_scores', [])
        if len(cv_scores) > 0:
            report.append("VALIDACIÓN CRUZADA:")
            report.append(f"   - Número de pliegues: {len(cv_scores)}")
            report.append(f"   - Scores por pliegue: {[f'{s:.4f}' for s in cv_scores]}")
            report.append(f"   - Promedio: {cv_scores.mean():.4f}")
            report.append(f"   - Desviación estándar: ± {cv_scores.std():.4f}")
            report.append("")
        
        # Resultados de evaluación
        report.append("RESULTADOS DE EVALUACIÓN:")
        report.append(f"   - Precisión conjunto prueba interno: {self.results.get('test_accuracy', 0):.4f}")
        report.append(f"   - F1-Score macro: {self.results.get('f1_macro', 0):.4f}")
        report.append(f"   - Precisión imágenes propias: {self.results.get('custom_accuracy', 0):.4f}")
        report.append("")
        
        # Análisis crítico
        report.append("ANÁLISIS CRÍTICO:")
        report.append("FORTALEZAS:")
        report.append("Buen desempeño en dataset estándar")
        report.append("Modelo robusto y generalizable")
        report.append("Optimización exitosa de hiperparámetros")
        report.append("")
        report.append("  LIMITACIONES:")
        report.append("Resolución limitada (8x8 píxeles)")
        report.append(" Dependiente de calidad de imagen de entrada")
        report.append(" Sensible a estilos de escritura diferentes")
        report.append("")
        
        # Conclusiones
        report.append(" CONCLUSIONES:")
        report.append("   El sistema Random Forest muestra excelente desempeño en el")
        report.append("   dataset estándar y capacidad de generalización aceptable")
        report.append("   a imágenes propias, con margen de mejora en preprocesamiento.")
        
        return "\n".join(report)
        
    def run_complete_pipeline(self):
        """Ejecutar el pipeline completo"""
        print_banner("SISTEMA DE CLASIFICACIÓN DE DÍGITOS MANUSCRITOS")
        print(f"Iniciando pipeline completo...")
        print(f"Tiempo de inicio: {datetime.now().strftime('%H:%M:%S')}")
        print()
        
        try:
            # Ejecutar todos los pasos
            self.setup_environment()
            image_names = self.load_and_prepare_data()
            self.train_base_model()
            self.optimize_model()
            self.evaluate_model()
            self.analyze_errors()
            self.generate_visualizations(image_names)
            self.save_results()
            self.generate_final_report()
            
            print("\n¡PIPELINE COMPLETADO EXITOSAMENTE!")
            print(f"Tiempo de finalización: {datetime.now().strftime('%H:%M:%S')}")
            print(f"Resultados guardados en: {self.config.RESULTS_DIR}")
            
            return self.best_model, self.results
            
        except Exception as e:
            print(f"\nERROR durante la ejecución: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None


def main():
    """
    Función principal - Punto de entrada del programa
    """
    # Crear y ejecutar pipeline
    pipeline = DigitClassificationPipeline()
    modelo, resultados = pipeline.run_complete_pipeline()
    
    if modelo is not None:
        print(f"\nRESUMEN FINAL:")
        print(f" Modelo entrenado y optimizado")
        print(f" Evaluaciones completadas")
        print(f" Visualizaciones generadas")
        print(f" Informe final creado")
        print(f"\nPara ver resultados detallados, revisa la carpeta 'results/'")
    else:
        print("\nEl pipeline no pudo completarse. Revisa los errores anteriores.")
    
    return modelo, resultados


if __name__ == "__main__":
    # Ejecutar programa principal
    model, results = main()
    
    # Mantener ventana abierta en algunos entornos
    input("\nPresiona Enter para salir...")