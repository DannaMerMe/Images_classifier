import os

class Config:
    """
    Configuración centralizada del proyecto
    """
    
    # Directorios principales
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    SRC_DIR = os.path.join(BASE_DIR, 'src')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    
    # Subdirectorios de datos
    DATA_RAW_DIR = os.path.join(DATA_DIR, 'raw')
    DATA_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
    
    # Subdirectorios de resultados
    RESULTS_MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
    RESULTS_PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
    RESULTS_METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
    RESULTS_REPORTS_DIR = os.path.join(RESULTS_DIR, 'reports')
    
    # Parámetros del modelo
    TEST_SIZE = 0.25
    RANDOM_STATE = 42
    CV_FOLDS = 5
    
    # Parámetros de Random Forest base
    RF_BASE_PARAMS = {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    # Grid para optimización
    RF_PARAM_GRID = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Formatos de imagen soportados
    IMAGE_EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    
    # Parámetros de visualización
    PLOT_STYLE = 'default'
    FIGURE_DPI = 300
    FIGURE_FORMAT = 'png'
