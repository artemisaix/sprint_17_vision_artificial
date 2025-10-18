# Guía de Testing y Validación

## Verificación de la Instalación

### 1. Verificar Requisitos del Sistema

```bash
# Verificar Python version (debe ser 3.8+)
python3 --version

# Verificar pip
pip --version
```

### 2. Instalar Dependencias

```bash
# Crear entorno virtual (recomendado)
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Verificar instalación
pip list | grep tensorflow
pip list | grep keras
```

### 3. Verificar Estructura del Proyecto

```bash
# El proyecto debe tener esta estructura:
ls -la
# Debe mostrar:
# - README.md
# - requirements.txt
# - .gitignore
# - age_estimation_model.ipynb
# - utils.py
# - predict.py
# - DATA_GUIDE.md
```

## Testing de Componentes

### Test 1: Verificar Importación de Utils

```python
# test_utils.py
import utils

# Verificar que todas las funciones están disponibles
assert hasattr(utils, 'load_labels')
assert hasattr(utils, 'create_data_generators')
assert hasattr(utils, 'evaluate_age_classification')
assert hasattr(utils, 'calculate_regression_metrics')
assert hasattr(utils, 'analyze_by_age_range')
assert hasattr(utils, 'preprocess_single_image')
assert hasattr(utils, 'predict_age')
assert hasattr(utils, 'print_evaluation_report')

print("✓ Todas las funciones de utils están disponibles")
```

### Test 2: Verificar Script de Predicción

```python
# test_predict.py
import sys
import subprocess

# Verificar que el script tiene help
result = subprocess.run(
    ['python', 'predict.py', '--help'],
    capture_output=True,
    text=True
)

assert 'Predice la edad' in result.stdout or 'usage:' in result.stderr
print("✓ Script de predicción tiene documentación")
```

### Test 3: Verificar Notebook

```python
# test_notebook.py
import json

with open('age_estimation_model.ipynb') as f:
    notebook = json.load(f)

# Verificar estructura básica
assert 'cells' in notebook
assert 'metadata' in notebook
assert len(notebook['cells']) > 0

# Verificar que hay celdas de código y markdown
cell_types = [cell['cell_type'] for cell in notebook['cells']]
assert 'code' in cell_types
assert 'markdown' in cell_types

print(f"✓ Notebook válido con {len(notebook['cells'])} celdas")
```

### Test 4: Verificar Funciones de Utils (con datos sintéticos)

```python
# test_utils_functions.py
import numpy as np
from utils import calculate_regression_metrics, evaluate_age_classification

# Crear datos sintéticos
np.random.seed(42)
true_ages = np.random.randint(10, 60, 100)
predicted_ages = true_ages + np.random.normal(0, 5, 100)

# Test métricas de regresión
metrics = calculate_regression_metrics(true_ages, predicted_ages)
assert 'mae' in metrics
assert 'mse' in metrics
assert 'rmse' in metrics
assert metrics['mae'] >= 0
print(f"✓ Métricas de regresión: MAE={metrics['mae']:.2f}")

# Test clasificación
class_metrics = evaluate_age_classification(true_ages, predicted_ages)
assert 'accuracy' in class_metrics
assert 'precision' in class_metrics
assert 'recall' in class_metrics
assert 0 <= class_metrics['accuracy'] <= 1
print(f"✓ Métricas de clasificación: Accuracy={class_metrics['accuracy']:.2%}")
```

## Testing con Jupyter Notebook

### Verificar que Jupyter está instalado

```bash
jupyter --version
# Debe mostrar la versión de jupyter

jupyter notebook --version
# Debe mostrar la versión de notebook
```

### Iniciar Jupyter

```bash
jupyter notebook
# Debe abrir el navegador con la interfaz de Jupyter
```

### Verificar el Notebook

1. Abrir `age_estimation_model.ipynb`
2. Verificar que todas las celdas están visibles
3. Intentar ejecutar la primera celda (importaciones)
4. Verificar que no hay errores de sintaxis

## Testing del Flujo Completo (con datos reales)

### Preparación

1. **Obtener un dataset de prueba:**
   ```bash
   # Descargar un dataset pequeño (por ejemplo, UTKFace sample)
   # O crear un dataset sintético para pruebas
   ```

2. **Organizar datos:**
   ```bash
   mkdir -p datasets/faces/final_files
   # Copiar imágenes a datasets/faces/final_files/
   # Crear labels.csv
   ```

3. **Validar dataset:**
   ```python
   import pandas as pd
   from pathlib import Path
   
   # Verificar estructura
   assert Path('datasets/faces/labels.csv').exists()
   assert Path('datasets/faces/final_files').exists()
   
   # Cargar y verificar CSV
   df = pd.read_csv('datasets/faces/labels.csv')
   assert 'file_name' in df.columns
   assert 'real_age' in df.columns
   assert len(df) > 0
   
   print(f"✓ Dataset válido con {len(df)} registros")
   ```

### Ejecutar Notebook Completo

1. Abrir el notebook en Jupyter
2. Seleccionar "Cell" → "Run All"
3. Esperar a que todas las celdas se ejecuten
4. Verificar que no hay errores

### Verificar Salidas Esperadas

El notebook debe generar:

1. **Visualizaciones:**
   - Distribución de edades
   - Curvas de entrenamiento
   - Gráficos de evaluación

2. **Métricas:**
   - MAE, MSE, RMSE
   - Accuracy, Precision, Recall
   - Análisis por rangos de edad

3. **Modelos guardados:**
   - `best_age_model.h5`
   - `age_estimation_model_final.h5`
   - `age_estimation_model_savedmodel/`

### Testing del Script de Predicción

```bash
# Después de entrenar el modelo
python predict.py datasets/faces/final_files/001.jpg --model best_age_model.h5

# Debe mostrar:
# - Edad estimada
# - Clasificación (menor/mayor)
# - Advertencias si está cerca del umbral
```

## Checklist de Validación

### Estructura del Proyecto
- [ ] README.md completo y bien documentado
- [ ] requirements.txt con todas las dependencias
- [ ] .gitignore configurado correctamente
- [ ] age_estimation_model.ipynb estructurado
- [ ] utils.py con funciones auxiliares
- [ ] predict.py funcional
- [ ] DATA_GUIDE.md con instrucciones claras

### Código
- [ ] Todos los archivos Python tienen sintaxis válida
- [ ] El notebook tiene estructura JSON válida
- [ ] No hay imports faltantes
- [ ] Las funciones tienen docstrings
- [ ] El código sigue buenas prácticas

### Funcionalidad
- [ ] Se puede instalar el entorno
- [ ] El notebook se puede abrir en Jupyter
- [ ] Las celdas del notebook se pueden ejecutar (con datos)
- [ ] El modelo se puede entrenar
- [ ] El modelo se puede evaluar
- [ ] El script de predicción funciona

### Documentación
- [ ] README explica el propósito del proyecto
- [ ] README tiene instrucciones de instalación
- [ ] README explica cómo usar el modelo
- [ ] DATA_GUIDE explica el formato de datos
- [ ] El código tiene comentarios adecuados

## Problemas Comunes y Soluciones

### Error: "No module named 'tensorflow'"
**Solución:** Instalar dependencias con `pip install -r requirements.txt`

### Error: "No se encontró el archivo labels.csv"
**Solución:** Crear el dataset siguiendo las instrucciones en DATA_GUIDE.md

### Error: GPU no disponible
**Solución:** El modelo puede entrenarse en CPU, aunque será más lento. Asegúrate de tener suficiente RAM.

### Error: "Kernel died" en Jupyter
**Solución:** Reducir BATCH_SIZE en el notebook o aumentar la memoria disponible

### Error al cargar imágenes
**Solución:** Verificar que las rutas en labels.csv coincidan con los nombres de archivo reales

## Métricas de Éxito

Para considerar el proyecto exitoso:

1. **Instalación:** Todo se instala sin errores
2. **Notebook:** Se ejecuta completamente sin errores
3. **Modelo:** Entrena y converge (loss disminuye)
4. **Evaluación:** MAE < 10 años (depende del dataset)
5. **Predicción:** El script funciona con una imagen de prueba
6. **Documentación:** Toda la documentación es clara y completa

## Siguiente Paso

Una vez validados todos los componentes, el proyecto está listo para:
- Entrenar con dataset completo
- Optimizar hiperparámetros
- Desplegar en producción
- Integrar con sistemas existentes
