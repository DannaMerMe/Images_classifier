# Images_classifier
# README.md - Documentación del Proyecto

# Sistema de Clasificación de imagenes

## Descripción del Proyecto

Este proyecto implementa un sistema completo de clasificación de dígitos manuscritos utilizando **Random Forest**, optimizado mediante validación cruzada y evaluado tanto en el dataset estándar `load_digits` como en imágenes personalizadas del usuario.

### Objetivos 
- **Entrenar** un modelo Random Forest para clasificación de dígitos (0-9)
- **Optimizar** hiperparámetros usando validación cruzada de 5 pliegues
- **Evaluar** el rendimiento en imágenes personalizadas del usuario
- **Generar** métricas detalladas: matriz de confusión, F1-score, precisión por clase
- **Analizar** errores frecuentes y limitaciones del modelo

---

##Estructura del Proyecto

```
images_classifier/
│
├── main.py                          
├── requirements.txt                
├── README.md                        
│
├── src/                            
│   ├── __init__.py
│   ├── models/                     
│   │   ├── __init__.py
│   │   ├── random_forest_classifier.py
│   │   └── model_optimizer.py
│   ├── data_processing/            
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── image_processor.py
│   ├── evaluation/                
│   │   ├── __init__.py
│   │   ├── metrics_calculator.py
│   │   └── visualization.py
│   └── utils/                      
│       ├── __init__.py
│       ├── config.py
│       └── helpers.py
│
├── data/                                      
│   │               
│   │  
│   │   
│   │
│   └── processed/    
│
└── results/                        
    ├── models/                     
    ├── plots/                      
    ├── metrics/                    
    └── reports/                   
    ```

---

## Instalación y Uso Rápido

### 1. **Instalar Dependencias**
```bash
pip install numpy matplotlib seaborn scikit-learn opencv-python pandas Pillow
```

### 2. **Preparar Imágenes**
- Coloca tus imágenes de dígitos en `data/raw/`
- **IMPORTANTE**: El nombre debe contener el dígito:
  - `digit_0_sample.png`
  - `imagen_del_5.jpg`
  - `mi_numero_3.png`
  - `imagen.png` (sin dígito en nombre)

### 3. **Ejecutar Proyecto**
```bash
python main.py
```

## Requisitos para Imágenes Personalizadas

### Formatos Soportados
- **PNG, JPG, JPEG, BMP, TIFF**

### Requisitos de Contenido
- **Un dígito por imagen** (0-9)
- **Resolución mínima**: 50×50 píxeles
- **Contraste**: Texto oscuro sobre fondo claro (o viceversa)
- **Centrado**: El dígito debe estar centrado en la imagen

### Nomenclatura Obligatoria
El sistema extrae la etiqueta del **nombre del archivo**:
```
CORRECTO:
   - digit_0_sample.png → etiqueta: 0
   - imagen_del_5.jpg → etiqueta: 5
   - numero_3_escrito.png → etiqueta: 3
   - 7_manuscrito.bmp → etiqueta: 7
```

---

##Métricas Evaluadas

### Validación Cruzada (K-Fold)
- **5 pliegues** de validación cruzada
- **Score promedio** y desviación estándar
- **Análisis de consistencia** entre pliegues

### Métricas de Clasificación
- **Precisión (Accuracy)**: Proporción de predicciones correctas
- **F1-Score Macro**: Promedio armónico de precisión y recall
- **F1-Score Weighted**: F1-score ponderado por clase
- **Matriz de Confusión**: Errores detallados por clase
- **Reporte por Clase**: Precision, recall, F1-score para cada dígito

### Conjuntos de Evaluación
1. **Prueba Interna**: 25% del dataset load_digits (450 muestras aprox.)
2. **Imágenes Propias**: Tu dataset personalizado

---

## Configuración Técnica del Modelo

### Random Forest - Parámetros Base
```python
{
    'n_estimators': 100,        # Balance precisión-velocidad
    'max_depth': 15,            # Previene overfitting
    'min_samples_split': 5,     # Robustez en divisiones
    'min_samples_leaf': 2,      # Mejor generalización
    'random_state': 42,         # Reproducibilidad
    'n_jobs': -1,              # Paralelización máxima
    'class_weight': 'balanced'  # Manejo de clases desbalanceadas
}
```

### Grid de Optimización
- **n_estimators**: [50, 100, 200]
- **max_depth**: [10, 15, 20, None]
- **min_samples_split**: [2, 5, 10]
- **min_samples_leaf**: [1, 2, 4]
- **max_features**: ['sqrt', 'log2']

---

##Interpretación de Resultados

### Scores de Validación Cruzada
- **> 0.95**: Excelente rendimiento
- **0.90-0.95**: Muy buen rendimiento
- **0.85-0.90**: Buen rendimiento
- **< 0.85**: Necesita mejoras

### Matriz de Confusión
- **Diagonal principal**: Predicciones correctas
- **Fuera diagonal**: Confusiones entre dígitos
- **Patrones comunes**: 6↔8, 4↔9, 3↔8

### F1-Score por Clase
- **Identifica** qué dígitos son más difíciles de clasificar
- **Balances** entre precisión y recall
- **Útil** para detectar clases problemáticas

---

## Archivos Generados

### En `results/models/`
- `best_random_forest_model.pkl`: Modelo optimizado
- `best_random_forest_model_metadata.json`: Metadatos

### En `results/plots/`
- `custom_images.png`: Visualización de tus imágenes
- `confusion_matrix_internal.png`: Matriz conjunto prueba
- `confusion_matrix_custom.png`: Matriz imágenes propias
- `cv_scores.png`: Gráfico validación cruzada

### En `results/metrics/`
- `evaluation_metrics.json`: Todas las métricas calculadas

### En `results/reports/`
- `informe_final.txt`: Reporte completo del análisis

---

## Análisis de Errores Implementado

### Errores Frecuentes
- **Identifica** qué dígitos se confunden más
- **Cuantifica** frecuencia de cada tipo de error
- **Analiza** patrones en imágenes propias vs dataset

### Dígitos Problemáticos
- **Ranking** de dígitos más difíciles de clasificar
- **Análisis** de por qué ciertos dígitos fallan
- **Sugerencias** de mejora específicas

---

## Limitaciones Conocidas

### Del Algoritmo
- **Resolución**: Limitado a 8×8 píxeles
- **Preprocesamiento