import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import pandas as pd

class Visualizer:
    """
    Clase para generar visualizaciones de los resultados
    """
    
    def __init__(self, style='default', figsize=(10, 8), dpi=300):
        """
        Inicializar visualizador
        
        Args:
            style: Estilo de matplotlib
            figsize: Tamaño por defecto de figuras
            dpi: DPI para guardar imágenes
        """
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        
        # Configurar matplotlib
        plt.style.use(style)
        sns.set_palette("husl")
        
        print(f"🎨 Visualizador inicializado con estilo '{style}'")
    
    def plot_custom_images(self, images, labels, image_names, save_path=None):
        """
        Visualizar imágenes personalizadas procesadas
        
        Args:
            images: Array de imágenes procesadas
            labels: Etiquetas de las imágenes
            image_names: Nombres de las imágenes
            save_path: Ruta para guardar la figura (opcional)
        """
        if len(images) == 0:
            print("No hay imágenes para visualizar")
            return
        
        print(f"Visualizando {len(images)} imágenes personalizadas...")
        
        # Calcular layout
        n_images = len(images)
        cols = min(5, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        
        # Manejar caso de una sola imagen
        if n_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        for i in range(n_images):
            # Convertir vector 1D a imagen 8x8
            img_2d = images[i].reshape(8, 8)
            
            # Mostrar imagen
            axes[i].imshow(img_2d, cmap='gray', interpolation='nearest')
            axes[i].set_title(f'{image_names[i]}\nEtiqueta: {labels[i]}', fontsize=10)
            axes[i].axis('off')
        
        # Ocultar ejes sobrantes
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Imágenes Personalizadas Procesadas (8×8 píxeles)', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Visualización guardada: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Matriz de Confusión", 
                             save_path=None, normalize=False):
        """
        Graficar matriz de confusión
        
        Args:
            y_true: Etiquetas reales
            y_pred: Predicciones
            title: Título del gráfico
            save_path: Ruta para guardar (opcional)
            normalize: Normalizar matriz
        """
        print(f"Generando {title.lower()}...")
        
        # Calcular matriz de confusión
        cm = confusion_matrix(y_true, y_pred, labels=range(10))
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            cbar_label = 'Proporción'
        else:
            fmt = 'd'
            cbar_label = 'Cantidad'
        
        # Crear figura
        plt.figure(figsize=(10, 8))
        
        # Heatmap
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=range(10), yticklabels=range(10),
                   cbar_kws={'label': cbar_label})
        
        plt.title(title, fontsize=16)
        plt.xlabel('Predicción', fontsize=12)
        plt.ylabel('Etiqueta Real', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Matriz de confusión guardada: {save_path}")
        
        plt.show()
    
    def plot_cross_validation_scores(self, cv_scores, save_path=None):
        """
        Graficar scores de validación cruzada
        
        Args:
            cv_scores: Array con scores de CV
            save_path: Ruta para guardar (opcional)
        """
        print("Graficando scores de validación cruzada...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico de barras con scores individuales
        folds = range(1, len(cv_scores) + 1)
        bars = ax1.bar(folds, cv_scores, alpha=0.7, color='skyblue', edgecolor='navy')
        
        # Línea con promedio
        mean_score = np.mean(cv_scores)
        ax1.axhline(y=mean_score, color='red', linestyle='--', 
                   label=f'Promedio: {mean_score:.4f}')
        
        # Configurar primer subplot
        ax1.set_xlabel('Pliegue')
        ax1.set_ylabel('Precisión')
        ax1.set_title('Scores por Pliegue de Validación Cruzada')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Agregar valores en las barras
        for bar, score in zip(bars, cv_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.4f}', ha='center', va='bottom')
        
        # Histograma de distribución
        ax2.hist(cv_scores, bins=max(3, len(cv_scores)//2), 
                alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        ax2.axvline(mean_score, color='red', linestyle='--', 
                   label=f'Promedio: {mean_score:.4f}')
        ax2.axvline(mean_score - np.std(cv_scores), color='orange', linestyle=':', 
                   label=f'±1 std: {np.std(cv_scores):.4f}')
        ax2.axvline(mean_score + np.std(cv_scores), color='orange', linestyle=':')
        
        ax2.set_xlabel('Precisión')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución de Scores CV')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Gráfico CV guardado: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_importances, save_path=None, top_n=20):
        """
        Visualizar importancia de características como mapa de calor
        
        Args:
            feature_importances: Array con importancias (64 elementos)
            save_path: Ruta para guardar (opcional)
            top_n: Número de características más importantes a resaltar
        """
        print(f"Visualizando importancia de características (top {top_n})...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Mapa de calor 8x8
        importance_map = feature_importances.reshape(8, 8)
        im1 = ax1.imshow(importance_map, cmap='YlOrRd', interpolation='nearest')
        ax1.set_title('Importancia de Píxeles (8×8)')
        ax1.set_xlabel('Columna')
        ax1.set_ylabel('Fila')
        
        # Agregar colorbar
        plt.colorbar(im1, ax=ax1, label='Importancia')
        
        # Agregar valores en el mapa
        for i in range(8):
            for j in range(8):
                ax1.text(j, i, f'{importance_map[i, j]:.3f}', 
                        ha='center', va='center', fontsize=8)
        
        # Gráfico de barras con top características
        sorted_indices = np.argsort(feature_importances)[::-1]
        top_importances = feature_importances[sorted_indices[:top_n]]
        top_positions = [f'({idx//8},{idx%8})' for idx in sorted_indices[:top_n]]
        
        bars = ax2.barh(range(top_n), top_importances, color='coral')
        ax2.set_yticks(range(top_n))
        ax2.set_yticklabels(top_positions)
        ax2.set_xlabel('Importancia')
        ax2.set_title(f'Top {top_n} Píxeles Más Importantes')
        ax2.invert_yaxis()
        
        # Agregar valores en las barras
        for i, (bar, importance) in enumerate(zip(bars, top_importances)):
            ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{importance:.4f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Importancia de características guardada: {save_path}")
        
        plt.show()
    
    def plot_prediction_confidence(self, predictions, probabilities, true_labels, save_path=None):
        """
        Visualizar distribución de confianza en predicciones
        
        Args:
            predictions: Predicciones del modelo
            probabilities: Probabilidades por clase
            true_labels: Etiquetas reales
            save_path: Ruta para guardar (opcional)
        """
        print("Visualizando distribución de confianza...")
        
        # Calcular confianza (probabilidad máxima)
        confidence = np.max(probabilities, axis=1)
        correct_mask = predictions == true_labels
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Histograma de confianza general
        ax1.hist(confidence, bins=20, alpha=0.7, color='skyblue', edgecolor='navy')
        ax1.axvline(np.mean(confidence), color='red', linestyle='--', 
                   label=f'Promedio: {np.mean(confidence):.3f}')
        ax1.set_xlabel('Confianza')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title('Distribución de Confianza en Predicciones')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Confianza por correcta vs incorrecta
        correct_conf = confidence[correct_mask]
        incorrect_conf = confidence[~correct_mask]
        
        ax2.hist([correct_conf, incorrect_conf], bins=15, alpha=0.7, 
                label=['Correctas', 'Incorrectas'], color=['green', 'red'])
        ax2.set_xlabel('Confianza')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Confianza: Predicciones Correctas vs Incorrectas')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Box plot de confianza por dígito
        confidence_by_digit = [confidence[true_labels == digit] for digit in range(10)]
        ax3.boxplot(confidence_by_digit, labels=range(10))
        ax3.set_xlabel('Dígito')
        ax3.set_ylabel('Confianza')
        ax3.set_title('Distribución de Confianza por Dígito')
        ax3.grid(True, alpha=0.3)
        
        # Scatter plot: confianza vs precisión
        digit_accuracy = []
        digit_mean_confidence = []
        
        for digit in range(10):
            digit_mask = true_labels == digit
            if np.sum(digit_mask) > 0:
                digit_acc = np.mean(predictions[digit_mask] == true_labels[digit_mask])
                digit_conf = np.mean(confidence[digit_mask])
                digit_accuracy.append(digit_acc)
                digit_mean_confidence.append(digit_conf)
            else:
                digit_accuracy.append(0)
                digit_mean_confidence.append(0)
        
        scatter = ax4.scatter(digit_mean_confidence, digit_accuracy, 
                            s=100, alpha=0.7, c=range(10), cmap='tab10')
        
        # Agregar etiquetas de dígitos
        for i, (conf, acc) in enumerate(zip(digit_mean_confidence, digit_accuracy)):
            ax4.annotate(str(i), (conf, acc), xytext=(5, 5), 
                        textcoords='offset points', fontsize=12)
        
        ax4.set_xlabel('Confianza Promedio')
        ax4.set_ylabel('Precisión')
        ax4.set_title('Confianza vs Precisión por Dígito')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Análisis de confianza guardado: {save_path}")
        
        plt.show()
    
    def plot_learning_curves(self, train_sizes, train_scores, val_scores, save_path=None):
        """
        Graficar curvas de aprendizaje
        
        Args:
            train_sizes: Tamaños de conjunto de entrenamiento
            train_scores: Scores de entrenamiento
            val_scores: Scores de validación
            save_path: Ruta para guardar (opcional)
        """
        print("📈 Graficando curvas de aprendizaje...")
        
        # Calcular medias y desviaciones estándar
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        
        # Curvas de entrenamiento
        plt.plot(train_sizes, train_mean, 'o-', color='blue', 
                label='Score Entrenamiento')
        plt.fill_between(train_sizes, train_mean - train_std,
                        train_mean + train_std, alpha=0.1, color='blue')
        
        # Curvas de validación
        plt.plot(train_sizes, val_mean, 'o-', color='red',
                label='Score Validación')
        plt.fill_between(train_sizes, val_mean - val_std,
                        val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel('Tamaño del Conjunto de Entrenamiento')
        plt.ylabel('Precisión')
        plt.title('Curvas de Aprendizaje')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Curvas de aprendizaje guardadas: {save_path}")
        
        plt.show()
    
    def create_comparison_plot(self, metrics_dict, save_path=None):
        """
        Crear gráfico comparativo de métricas
        
        Args:
            metrics_dict: Diccionario con métricas de diferentes evaluaciones
            save_path: Ruta para guardar (opcional)
        """
        print("Creando gráfico comparativo de métricas...")
        
        # Extraer métricas para comparación
        categories = []
        accuracies = []
        f1_scores = []
        
        if 'test_accuracy' in metrics_dict:
            categories.append('Prueba Interna')
            accuracies.append(metrics_dict['test_accuracy'])
            f1_scores.append(metrics_dict.get('f1_macro', 0))
        
        if 'custom_accuracy' in metrics_dict:
            categories.append('Imágenes Propias')
            accuracies.append(metrics_dict['custom_accuracy'])
            f1_scores.append(metrics_dict.get('custom_f1_macro', 0))
        
        if len(categories) == 0:
            print("No hay métricas para comparar")
            return
        
        # Crear gráfico de barras
        x = np.arange(len(categories))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, accuracies, width, label='Precisión', 
                      color='skyblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score Macro', 
                      color='lightcoral', alpha=0.8)
        
        # Configurar gráfico
        ax.set_xlabel('Conjunto de Datos')
        ax.set_ylabel('Score')
        ax.set_title('Comparación de Métricas por Conjunto de Datos')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Agregar valores en las barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Gráfico comparativo guardado: {save_path}")
        
        plt.show()