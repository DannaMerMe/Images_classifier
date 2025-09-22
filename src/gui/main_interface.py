import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
import numpy as np
import os
import threading
from PIL import Image, ImageTk

# Importar módulos del proyecto
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from data_processing.data_loader import DataLoader
from data_processing.image_processor import ImageProcessor
from models.random_forest_classifier import RandomForestClassifier
from models.model_optimizer import ModelOptimizer
from evaluation.metrics_calculator import MetricsCalculator
from evaluation.visualization import Visualizer

class DigitsClassifierApp:
    """
    Interfaz gráfica principal para el clasificador de dígitos
    """
    
    def __init__(self, root):
        self.root = root
        self.setup_window()
        
        # Inicializar componentes del proyecto
        self.data_loader = DataLoader()
        self.image_processor = ImageProcessor()
        self.model = None
        self.optimizer = ModelOptimizer()
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer()
        
        # Variables de datos
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.custom_images = None
        self.custom_labels = None
        self.custom_names = None
        
        # Variables de estado
        self.model_trained = False
        self.data_loaded = False
        
        self.create_widgets()
        
    def setup_window(self):
        """Configurar ventana principal"""
        self.root.title("Clasificador de Dígitos - Machine Learning")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Configurar estilo
        style = ttk.Style()
        style.theme_use('clam')
        
    def create_widgets(self):
        """Crear widgets de la interfaz"""
        # Crear notebook (pestañas)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Pestaña 1: Cargar datos
        self.create_data_tab()
        
        # Pestaña 2: Entrenar modelo
        self.create_training_tab()
        
        # Pestaña 3: Imágenes personalizadas
        self.create_custom_images_tab()
        
        # Pestaña 4: Evaluación
        self.create_evaluation_tab()
        
        # Pestaña 5: Predicciones
        self.create_prediction_tab()
        
    def create_data_tab(self):
        """Crear pestaña de carga de datos"""
        tab_data = ttk.Frame(self.notebook)
        self.notebook.add(tab_data, text="Cargar Datos")
        
        # Frame principal
        main_frame = ttk.Frame(tab_data)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Título
        title_label = ttk.Label(main_frame, text="Cargar Dataset de Dígitos", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Frame de controles
        controls_frame = ttk.LabelFrame(main_frame, text="Configuración")
        controls_frame.pack(fill='x', pady=(0, 20))
        
        # Tamaño de conjunto de prueba
        ttk.Label(controls_frame, text="Tamaño conjunto de prueba:").grid(
            row=0, column=0, sticky='w', padx=10, pady=10)
        
        self.test_size_var = tk.DoubleVar(value=0.25)
        test_size_scale = ttk.Scale(controls_frame, from_=0.1, to=0.5, 
                                   variable=self.test_size_var, orient='horizontal')
        test_size_scale.grid(row=0, column=1, sticky='ew', padx=10, pady=10)
        
        self.test_size_label = ttk.Label(controls_frame, text="25%")
        self.test_size_label.grid(row=0, column=2, padx=10, pady=10)
        
        test_size_scale.configure(command=self.update_test_size_label)
        
        # Normalización
        self.normalize_var = tk.BooleanVar()
        normalize_check = ttk.Checkbutton(controls_frame, text="Normalizar datos", 
                                         variable=self.normalize_var)
        normalize_check.grid(row=1, column=0, columnspan=3, sticky='w', padx=10, pady=10)
        
        controls_frame.columnconfigure(1, weight=1)
        
        # Botón de carga
        load_button = ttk.Button(main_frame, text="Cargar Dataset", 
                                command=self.load_dataset_thread)
        load_button.pack(pady=10)
        
        # Frame de información
        info_frame = ttk.LabelFrame(main_frame, text="Información del Dataset")
        info_frame.pack(fill='both', expand=True)
        
        self.info_text = tk.Text(info_frame, height=15, width=80, wrap='word',
                                font=('Consolas', 10))
        info_scrollbar = ttk.Scrollbar(info_frame, command=self.info_text.yview)
        self.info_text.config(yscrollcommand=info_scrollbar.set)
        
        self.info_text.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        info_scrollbar.pack(side='right', fill='y', pady=10)
        
        # Frame para gráficos de muestra
        self.sample_frame = ttk.LabelFrame(main_frame, text="Muestras del Dataset")
        self.sample_frame.pack(fill='both', expand=True, pady=(20, 0))
        
    def create_training_tab(self):
        """Crear pestaña de entrenamiento"""
        tab_training = ttk.Frame(self.notebook)
        self.notebook.add(tab_training, text="Entrenamiento")
        
        # Frame principal
        main_frame = ttk.Frame(tab_training)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Título
        title_label = ttk.Label(main_frame, text="Entrenamiento del Modelo", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Configuración del modelo
        config_frame = ttk.LabelFrame(main_frame, text="Configuración Random Forest")
        config_frame.pack(fill='x', pady=(0, 20))
        
        # Número de árboles
        ttk.Label(config_frame, text="Número de árboles:").grid(
            row=0, column=0, sticky='w', padx=10, pady=5)
        
        self.n_estimators_var = tk.IntVar(value=100)
        n_estimators_spin = ttk.Spinbox(config_frame, from_=10, to=500, 
                                       textvariable=self.n_estimators_var, width=10)
        n_estimators_spin.grid(row=0, column=1, sticky='w', padx=10, pady=5)
        
        # Profundidad máxima
        ttk.Label(config_frame, text="Profundidad máxima:").grid(
            row=1, column=0, sticky='w', padx=10, pady=5)
        
        self.max_depth_var = tk.IntVar(value=15)
        max_depth_spin = ttk.Spinbox(config_frame, from_=5, to=50, 
                                    textvariable=self.max_depth_var, width=10)
        max_depth_spin.grid(row=1, column=1, sticky='w', padx=10, pady=5)
        
        # Botones de entrenamiento
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill='x', pady=10)
        
        train_button = ttk.Button(buttons_frame, text="Entrenar Modelo Base", 
                                 command=self.train_model_thread)
        train_button.pack(side='left', padx=(0, 10))
        
        optimize_button = ttk.Button(buttons_frame, text="⚡ Optimizar Hiperparámetros", 
                                    command=self.optimize_model_thread)
        optimize_button.pack(side='left')
        
        # Barra de progreso
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill='x', pady=10)
        
        # Área de resultados
        results_frame = ttk.LabelFrame(main_frame, text="Resultados del Entrenamiento")
        results_frame.pack(fill='both', expand=True)
        
        self.training_text = tk.Text(results_frame, height=20, wrap='word',
                                    font=('Consolas', 10))
        training_scrollbar = ttk.Scrollbar(results_frame, command=self.training_text.yview)
        self.training_text.config(yscrollcommand=training_scrollbar.set)
        
        self.training_text.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        training_scrollbar.pack(side='right', fill='y', pady=10)
        
    def create_custom_images_tab(self):
        """Crear pestaña de imágenes personalizadas"""
        tab_custom = ttk.Frame(self.notebook)
        self.notebook.add(tab_custom, text="📸 Imágenes Propias")
        
        # Frame principal
        main_frame = ttk.Frame(tab_custom)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Título
        title_label = ttk.Label(main_frame, text="Procesar Imágenes Personalizadas", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Instrucciones
        instructions = """
        Instrucciones:
        1. Selecciona una carpeta que contenga imágenes de dígitos (0-9)
        2. Los nombres de archivo deben contener el dígito (ej: "digit_5.png", "numero_3.jpg")
        3. Las imágenes serán redimensionadas a 8x8 píxeles automáticamente
        4. Formatos soportados: PNG, JPG, JPEG, BMP, TIFF
        """
        
        inst_label = ttk.Label(main_frame, text=instructions, justify='left',
                              background='#f0f0f0', relief='ridge', padding=10)
        inst_label.pack(fill='x', pady=(0, 20))
        
        # Botón para seleccionar carpeta
        select_button = ttk.Button(main_frame, text="Seleccionar Carpeta de Imágenes", 
                                  command=self.select_images_folder)
        select_button.pack(pady=10)
        
        # Frame de información
        info_custom_frame = ttk.LabelFrame(main_frame, text="Imágenes Procesadas")
        info_custom_frame.pack(fill='both', expand=True, pady=(20, 0))
        
        self.custom_info_text = tk.Text(info_custom_frame, height=10, wrap='word',
                                       font=('Consolas', 10))
        custom_scrollbar = ttk.Scrollbar(info_custom_frame, command=self.custom_info_text.yview)
        self.custom_info_text.config(yscrollcommand=custom_scrollbar.set)
        
        self.custom_info_text.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        custom_scrollbar.pack(side='right', fill='y', pady=10)
        
        # Frame para visualización de imágenes
        self.custom_viz_frame = ttk.LabelFrame(main_frame, text="Visualización")
        self.custom_viz_frame.pack(fill='both', expand=True, pady=(20, 0))
        
    def create_evaluation_tab(self):
        """Crear pestaña de evaluación"""
        tab_eval = ttk.Frame(self.notebook)
        self.notebook.add(tab_eval, text="Evaluación")
        
        # Frame principal
        main_frame = ttk.Frame(tab_eval)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Título
        title_label = ttk.Label(main_frame, text="Evaluación del Modelo", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Botones de evaluación
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill='x', pady=10)
        
        eval_internal_button = ttk.Button(buttons_frame, text="Evaluar Conjunto Prueba", 
                                         command=self.evaluate_internal_test)
        eval_internal_button.pack(side='left', padx=(0, 10))
        
        eval_custom_button = ttk.Button(buttons_frame, text="Evaluar Imágenes Propias", 
                                       command=self.evaluate_custom_images)
        eval_custom_button.pack(side='left', padx=(0, 10))
        
        show_confusion_button = ttk.Button(buttons_frame, text="Matriz Confusión", 
                                          command=self.show_confusion_matrix)
        show_confusion_button.pack(side='left')
        
        # Área de métricas
        metrics_frame = ttk.LabelFrame(main_frame, text="Métricas de Rendimiento")
        metrics_frame.pack(fill='both', expand=True)
        
        self.metrics_text = tk.Text(metrics_frame, height=15, wrap='word',
                                   font=('Consolas', 10))
        metrics_scrollbar = ttk.Scrollbar(metrics_frame, command=self.metrics_text.yview)
        self.metrics_text.config(yscrollcommand=metrics_scrollbar.set)
        
        self.metrics_text.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        metrics_scrollbar.pack(side='right', fill='y', pady=10)
        
        # Frame para gráficos de evaluación
        self.eval_viz_frame = ttk.LabelFrame(main_frame, text="Visualizaciones")
        self.eval_viz_frame.pack(fill='both', expand=True, pady=(20, 0))
        
    def create_prediction_tab(self):
        """Crear pestaña de predicciones"""
        tab_pred = ttk.Frame(self.notebook)
        self.notebook.add(tab_pred, text="Predicciones")
        
        # Frame principal
        main_frame = ttk.Frame(tab_pred)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Título
        title_label = ttk.Label(main_frame, text="Realizar Predicciones", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Botón para predecir imágenes personalizadas
        predict_button = ttk.Button(main_frame, text="Predecir Imágenes Personalizadas", 
                                   command=self.predict_custom_images)
        predict_button.pack(pady=10)
        
        # Frame para resultados de predicción
        pred_results_frame = ttk.LabelFrame(main_frame, text="Resultados de Predicción")
        pred_results_frame.pack(fill='both', expand=True)
        
        self.prediction_text = tk.Text(pred_results_frame, height=10, wrap='word',
                                      font=('Consolas', 10))
        pred_scrollbar = ttk.Scrollbar(pred_results_frame, command=self.prediction_text.yview)
        self.prediction_text.config(yscrollcommand=pred_scrollbar.set)
        
        self.prediction_text.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        pred_scrollbar.pack(side='right', fill='y', pady=10)
        
        # Frame para visualización de predicciones
        self.pred_viz_frame = ttk.LabelFrame(main_frame, text="Visualización de Predicciones")
        self.pred_viz_frame.pack(fill='both', expand=True, pady=(20, 0))
    
    def update_test_size_label(self, value):
        """Actualizar etiqueta del tamaño de prueba"""
        self.test_size_label.config(text=f"{float(value)*100:.0f}%")
    
    def log_to_text_widget(self, text_widget, message):
        """Agregar mensaje a widget de texto"""
        text_widget.insert(tk.END, message + "\n")
        text_widget.see(tk.END)
        self.root.update()
    
    def clear_text_widget(self, text_widget):
        """Limpiar widget de texto"""
        text_widget.delete(1.0, tk.END)
    
    def load_dataset_thread(self):
        """Cargar dataset en hilo separado"""
        thread = threading.Thread(target=self.load_dataset)
        thread.daemon = True
        thread.start()
    
    def load_dataset(self):
        """Cargar dataset de dígitos"""
        try:
            self.clear_text_widget(self.info_text)
            self.log_to_text_widget(self.info_text, "Cargando dataset de dígitos...")
            
            # Cargar datos
            self.X_train, self.X_test, self.y_train, self.y_test = self.data_loader.load_digits_dataset(
                test_size=self.test_size_var.get(),
                normalize=self.normalize_var.get()
            )
            
            self.data_loaded = True
            
            # Mostrar información
            self.log_to_text_widget(self.info_text, f"Dataset cargado exitosamente!")
            self.log_to_text_widget(self.info_text, f"Información del dataset:")
            self.log_to_text_widget(self.info_text, f"   - Entrenamiento: {self.X_train.shape[0]} muestras")
            self.log_to_text_widget(self.info_text, f"   - Prueba: {self.X_test.shape[0]} muestras")
            self.log_to_text_widget(self.info_text, f"   - Dimensiones: {self.X_train.shape[1]} (8x8 píxeles)")
            
            # Obtener distribución de clases
            distribution = self.data_loader.get_class_distribution()
            self.log_to_text_widget(self.info_text, "\nDistribución por dígito:")
            for digit, count in distribution.items():
                self.log_to_text_widget(self.info_text, f"   Dígito {digit}: {count} muestras")
            
            # Mostrar muestras
            self.show_sample_images()
            
            messagebox.showinfo("Éxito", "Dataset cargado correctamente")
            
        except Exception as e:
            self.log_to_text_widget(self.info_text, f"Error: {str(e)}")
            messagebox.showerror("Error", f"Error cargando dataset: {str(e)}")
    
    def show_sample_images(self):
        """Mostrar imágenes de muestra"""
        try:
            # Limpiar frame anterior
            for widget in self.sample_frame.winfo_children():
                widget.destroy()
            
            # Obtener muestras
            sample_images, sample_labels = self.data_loader.get_sample_images(10)
            
            # Crear figura matplotlib
            fig, axes = plt.subplots(2, 5, figsize=(10, 4))
            fig.suptitle('Muestras del Dataset (8x8 píxeles)', fontsize=14)
            
            for i, (img, label) in enumerate(zip(sample_images, sample_labels)):
                row, col = i // 5, i % 5
                img_2d = img.reshape(8, 8)
                axes[row, col].imshow(img_2d, cmap='gray')
                axes[row, col].set_title(f'Dígito: {label}')
                axes[row, col].axis('off')
            
            plt.tight_layout()
            
            # Integrar con tkinter
            canvas = FigureCanvasTkAgg(fig, self.sample_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            self.log_to_text_widget(self.info_text, f"Error mostrando muestras: {str(e)}")
    
    def train_model_thread(self):
        """Entrenar modelo en hilo separado"""
        if not self.data_loaded:
            messagebox.showwarning("Advertencia", "Primero carga el dataset")
            return
            
        thread = threading.Thread(target=self.train_model)
        thread.daemon = True
        thread.start()
    
    def train_model(self):
        """Entrenar modelo Random Forest"""
        try:
            self.clear_text_widget(self.training_text)
            self.progress.start()
            
            self.log_to_text_widget(self.training_text, "Iniciando entrenamiento del modelo...")
            
            # Crear modelo con parámetros configurados
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators_var.get(),
                max_depth=self.max_depth_var.get()
            )
            
            # Entrenar modelo
            test_accuracy = self.model.train_base_model(
                self.X_train, self.y_train, 
                self.X_test, self.y_test
            )
            
            self.model_trained = True
            
            # Mostrar resultados
            self.log_to_text_widget(self.training_text, "Entrenamiento completado!")
            self.log_to_text_widget(self.training_text, f"Precisión en prueba: {test_accuracy:.4f}")
            
            # Obtener importancia de características
            self.log_to_text_widget(self.training_text, "\nCaracterísticas más importantes:")
            importance_dict = self.model.get_feature_importance(10)
            for idx, importance in importance_dict.items():
                row, col = idx // 8, idx % 8
                self.log_to_text_widget(self.training_text, 
                    f"   Píxel [{row},{col}]: {importance:.4f}")
            
            self.progress.stop()
            messagebox.showinfo("Éxito", "Modelo entrenado correctamente")
            
        except Exception as e:
            self.progress.stop()
            self.log_to_text_widget(self.training_text, f"Error: {str(e)}")
            messagebox.showerror("Error", f"Error entrenando modelo: {str(e)}")
    
    def optimize_model_thread(self):
        """Optimizar modelo en hilo separado"""
        if not self.data_loaded:
            messagebox.showwarning("Advertencia", "Primero carga el dataset")
            return
            
        thread = threading.Thread(target=self.optimize_model)
        thread.daemon = True
        thread.start()
    
    def optimize_model(self):
        """Optimizar hiperparámetros"""
        try:
            self.progress.start()
            self.log_to_text_widget(self.training_text, "\nOptimizando hiperparámetros...")
            
            # Optimizar modelo
            best_model, best_params, cv_scores = self.optimizer.optimize_random_forest(
                self.X_train, self.y_train, cv_folds=3, search_type='grid'
            )
            
            self.model = RandomForestClassifier(**best_params)
            self.model.model = best_model
            self.model.is_trained = True
            self.model.feature_importances = best_model.feature_importances_
            self.model_trained = True
            
            # Mostrar resultados
            self.log_to_text_widget(self.training_text, "Optimización completada!")
            self.log_to_text_widget(self.training_text, f"Mejores parámetros: {best_params}")
            self.log_to_text_widget(self.training_text, f"core CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            self.progress.stop()
            messagebox.showinfo("Éxito", "Modelo optimizado correctamente")
            
        except Exception as e:
            self.progress.stop()
            self.log_to_text_widget(self.training_text, f"Error: {str(e)}")
            messagebox.showerror("Error", f"Error optimizando modelo: {str(e)}")
    
    def select_images_folder(self):
        """Seleccionar carpeta de imágenes"""
        folder_path = filedialog.askdirectory(title="Seleccionar carpeta con imágenes de dígitos")
        
        if folder_path:
            self.process_custom_images(folder_path)
    
    def process_custom_images(self, folder_path):
        """Procesar imágenes personalizadas"""
        try:
            self.clear_text_widget(self.custom_info_text)
            self.log_to_text_widget(self.custom_info_text, f"Procesando imágenes desde: {folder_path}")
            
            # Procesar imágenes
            self.custom_images, self.custom_labels, self.custom_names = \
                self.image_processor.process_custom_images(folder_path)
            
            if len(self.custom_images) > 0:
                # Validar imágenes procesadas
                is_valid = self.image_processor.validate_processed_images(
                    self.custom_images, self.custom_labels
                )
                
                if is_valid:
                    self.log_to_text_widget(self.custom_info_text, 
                        f"{len(self.custom_images)} imágenes procesadas correctamente")
                    
                    # Mostrar distribución
                    self.log_to_text_widget(self.custom_info_text, "\nDistribución por dígito:")
                    unique, counts = np.unique(self.custom_labels, return_counts=True)
                    for digit, count in zip(unique, counts):
                        self.log_to_text_widget(self.custom_info_text, f"   Dígito {digit}: {count} imagen(es)")
                    
                    # Visualizar imágenes procesadas
                    self.visualize_custom_images()
                    
                else:
                    self.log_to_text_widget(self.custom_info_text, "Error en validación de imágenes")
            else:
                self.log_to_text_widget(self.custom_info_text, "No se procesaron imágenes")
                
        except Exception as e:
            self.log_to_text_widget(self.custom_info_text, f"Error: {str(e)}")
            messagebox.showerror("Error", f"Error procesando imágenes: {str(e)}")
    
    def visualize_custom_images(self):
        """Visualizar imágenes personalizadas procesadas"""
        if self.custom_images is None or len(self.custom_images) == 0:
            return
            
        try:
            # Limpiar frame anterior
            for widget in self.custom_viz_frame.winfo_children():
                widget.destroy()
            
            # Calcular layout
            n_images = len(self.custom_images)
            cols = min(5, n_images)
            rows = (n_images + cols - 1) // cols
            
            # Crear figura
            fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
            if n_images == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes if hasattr(axes, '__len__') else [axes]
            else:
                axes = axes.flatten()
            
            fig.suptitle('Imágenes Personalizadas Procesadas', fontsize=14)
            
            for i in range(n_images):
                img_2d = self.custom_images[i].reshape(8, 8)
                axes[i].imshow(img_2d, cmap='gray', interpolation='nearest')
                axes[i].set_title(f'{self.custom_names[i]}\nEtiqueta: {self.custom_labels[i]}', fontsize=8)
                axes[i].axis('off')
            
            # Ocultar ejes sobrantes
            for i in range(n_images, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            
            # Integrar con tkinter
            canvas = FigureCanvasTkAgg(fig, self.custom_viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            self.log_to_text_widget(self.custom_info_text, f"Error visualizando: {str(e)}")
    
    def evaluate_internal_test(self):
        """Evaluar modelo en conjunto de prueba interno"""
        if not self.model_trained:
            messagebox.showwarning("Advertencia", "Primero entrena el modelo")
            return
            
        try:
            self.clear_text_widget(self.metrics_text)
            self.log_to_text_widget(self.metrics_text, "Evaluando modelo en conjunto de prueba...")
            
            # Evaluar modelo
            metrics = self.metrics_calculator.evaluate_internal_test(
                self.model, self.X_test, self.y_test
            )
            
            # Mostrar resultados
            self.log_to_text_widget(self.metrics_text, "Evaluación completada!")
            self.log_to_text_widget(self.metrics_text, f"Precisión: {metrics['test_accuracy']:.4f}")
            self.log_to_text_widget(self.metrics_text, f"F1-Score macro: {metrics['f1_macro']:.4f}")
            self.log_to_text_widget(self.metrics_text, f"F1-Score weighted: {metrics['f1_weighted']:.4f}")
            self.log_to_text_widget(self.metrics_text, f"Muestras evaluadas: {metrics['n_samples']}")
            
            # Mostrar métricas por clase
            self.log_to_text_widget(self.metrics_text, "\nMétricas por dígito:")
            for digit, metrics_class in metrics['class_metrics'].items():
                self.log_to_text_widget(self.metrics_text, 
                    f"   Dígito {digit}: Precisión={metrics_class['precision']:.3f}, "
                    f"Recall={metrics_class['recall']:.3f}, F1={metrics_class['f1-score']:.3f}")
            
            # Guardar métricas para uso posterior
            self.current_metrics = metrics
            
        except Exception as e:
            self.log_to_text_widget(self.metrics_text, f"Error: {str(e)}")
            messagebox.showerror("Error", f"Error evaluando modelo: {str(e)}")
    
    def evaluate_custom_images(self):
        """Evaluar modelo con imágenes personalizadas"""
        if not self.model_trained:
            messagebox.showwarning("Advertencia", "Primero entrena el modelo")
            return
            
        if self.custom_images is None or len(self.custom_images) == 0:
            messagebox.showwarning("Advertencia", "Primero carga imágenes personalizadas")
            return
            
        try:
            self.log_to_text_widget(self.metrics_text, "\nEvaluando imágenes personalizadas...")
            
            # Evaluar imágenes personalizadas
            custom_metrics = self.metrics_calculator.evaluate_custom_images(
                self.model, self.custom_images, self.custom_labels
            )
            
            # Mostrar resultados
            self.log_to_text_widget(self.metrics_text, "Evaluación de imágenes personalizadas completada!")
            self.log_to_text_widget(self.metrics_text, f"Precisión: {custom_metrics['custom_accuracy']:.4f}")
            self.log_to_text_widget(self.metrics_text, f"F1-Score: {custom_metrics['custom_f1_macro']:.4f}")
            self.log_to_text_widget(self.metrics_text, f"Imágenes evaluadas: {custom_metrics['n_custom_samples']}")
            
            # Mostrar resultados detallados
            self.log_to_text_widget(self.metrics_text, "\nResultados detallados:")
            for result in custom_metrics['detailed_results']:
                status = "ok" if result['is_correct'] else "error"
                self.log_to_text_widget(self.metrics_text,
                    f"   {self.custom_names[result['image_index']]}: "
                    f"Real={result['true_label']}, Pred={result['predicted_label']}, "
                    f"Conf={result['confidence']:.3f} {status}")
            
        except Exception as e:
            self.log_to_text_widget(self.metrics_text, f"Error: {str(e)}")
            messagebox.showerror("Error", f"Error evaluando imágenes: {str(e)}")
    
    def show_confusion_matrix(self):
        """Mostrar matriz de confusión"""
        if not self.model_trained:
            messagebox.showwarning("Advertencia", "Primero entrena el modelo")
            return
            
        try:
            # Limpiar frame anterior
            for widget in self.eval_viz_frame.winfo_children():
                widget.destroy()
            
            # Generar predicciones
            y_pred = self.model.predict(self.X_test)
            
            # Crear matriz de confusión
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            
            cm = confusion_matrix(self.y_test, y_pred, labels=range(10))
            
            # Crear figura
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=range(10), yticklabels=range(10),
                       ax=ax)
            ax.set_title('Matriz de Confusión - Conjunto de Prueba')
            ax.set_xlabel('Predicción')
            ax.set_ylabel('Etiqueta Real')
            
            plt.tight_layout()
            
            # Integrar con tkinter
            canvas = FigureCanvasTkAgg(fig, self.eval_viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error mostrando matriz de confusión: {str(e)}")
    
    def predict_custom_images(self):
        """Realizar predicciones en imágenes personalizadas"""
        if not self.model_trained:
            messagebox.showwarning("Advertencia", "Primero entrena el modelo")
            return
            
        if self.custom_images is None or len(self.custom_images) == 0:
            messagebox.showwarning("Advertencia", "Primero carga imágenes personalizadas")
            return
            
        try:
            self.clear_text_widget(self.prediction_text)
            self.log_to_text_widget(self.prediction_text, "Realizando predicciones...")
            
            # Realizar predicciones
            predictions = self.model.predict(self.custom_images)
            probabilities = self.model.predict_proba(self.custom_images)
            
            # Mostrar resultados
            self.log_to_text_widget(self.prediction_text, "Predicciones completadas!")
            self.log_to_text_widget(self.prediction_text, "\n Resultados:")
            
            for i, (name, pred, prob) in enumerate(zip(self.custom_names, predictions, probabilities)):
                confidence = np.max(prob)
                true_label = self.custom_labels[i] if self.custom_labels is not None else "?"
                
                self.log_to_text_widget(self.prediction_text,
                    f"   {name}:")
                self.log_to_text_widget(self.prediction_text,
                    f"     Predicción: {pred} (confianza: {confidence:.3f})")
                
                if self.custom_labels is not None:
                    correct = "ok" if pred == true_label else "error"
                    self.log_to_text_widget(self.prediction_text,
                        f"     Real: {true_label} {correct}")
                
                # Top 3 probabilidades
                top_3_indices = np.argsort(prob)[-3:][::-1]
                self.log_to_text_widget(self.prediction_text, "     Top 3 probabilidades:")
                for idx in top_3_indices:
                    self.log_to_text_widget(self.prediction_text,
                        f"       Dígito {idx}: {prob[idx]:.3f}")
                self.log_to_text_widget(self.prediction_text, "")
            
            # Visualizar predicciones
            self.visualize_predictions(predictions, probabilities)
            
        except Exception as e:
            self.log_to_text_widget(self.prediction_text, f" Error: {str(e)}")
            messagebox.showerror("Error", f"Error realizando predicciones: {str(e)}")
    
    def visualize_predictions(self, predictions, probabilities):
        """Visualizar predicciones con imágenes"""
        try:
            # Limpiar frame anterior
            for widget in self.pred_viz_frame.winfo_children():
                widget.destroy()
            
            if len(self.custom_images) == 0:
                return
            
            # Calcular layout
            n_images = len(self.custom_images)
            cols = min(4, n_images)
            rows = (n_images + cols - 1) // cols
            
            # Crear figura
            fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
            if n_images == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes if hasattr(axes, '__len__') else [axes]
            else:
                axes = axes.flatten()
            
            fig.suptitle('Predicciones en Imágenes Personalizadas', fontsize=16)
            
            for i in range(n_images):
                img_2d = self.custom_images[i].reshape(8, 8)
                confidence = np.max(probabilities[i])
                
                # Color del borde según correctitud
                if self.custom_labels is not None:
                    is_correct = predictions[i] == self.custom_labels[i]
                    color = 'green' if is_correct else 'red'
                    title = f'Pred: {predictions[i]}, Real: {self.custom_labels[i]}\nConf: {confidence:.3f}'
                else:
                    color = 'blue'
                    title = f'Predicción: {predictions[i]}\nConfianza: {confidence:.3f}'
                
                axes[i].imshow(img_2d, cmap='gray', interpolation='nearest')
                axes[i].set_title(title, fontsize=10, color=color)
                axes[i].axis('off')
                
                # Agregar borde coloreado
                rect = patches.Rectangle((0, 0), 7, 7, linewidth=3, edgecolor=color, 
                                       facecolor='none', transform=axes[i].transData)
                axes[i].add_patch(rect)
            
            # Ocultar ejes sobrantes
            for i in range(n_images, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            
            # Integrar con tkinter
            canvas = FigureCanvasTkAgg(fig, self.pred_viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            self.log_to_text_widget(self.prediction_text, f"Error visualizando: {str(e)}")

def main():
    """Función principal para ejecutar la aplicación"""
    root = tk.Tk()
    app = DigitsClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()