# Sprint 17: Visión Artificial - Estimación de Edad

## 📋 Descripción del Proyecto

Good Seed busca evitar la venta de alcohol a menores de edad utilizando ciencia de datos. Las tiendas cuentan con cámaras que se activan automáticamente cuando se realiza una compra de alcohol. Este proyecto implementa un modelo de visión artificial que estima la edad de una persona a partir de fotografías.

## 🎯 Objetivo

Desarrollar un modelo de deep learning capaz de:
- Estimar la edad de una persona a partir de una fotografía facial
- Identificar con alta precisión si una persona es menor de 18 años
- Proporcionar una herramienta de apoyo para prevenir la venta de alcohol a menores

## 🏗️ Estructura del Proyecto

```
sprint_17_vision_artificial/
├── README.md                          # Este archivo
├── requirements.txt                    # Dependencias del proyecto
├── .gitignore                         # Archivos a ignorar en git
├── age_estimation_model.ipynb         # Notebook principal con el modelo
└── datasets/                          # Directorio para los datos (no incluido en git)
    └── faces/
        ├── labels.csv                 # Etiquetas con nombres de archivo y edades
        └── final_files/               # Imágenes de rostros
```

## 🚀 Configuración del Entorno

### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Jupyter Notebook o JupyterLab

### Instalación

1. **Clonar el repositorio:**
```bash
git clone https://github.com/artemisaix/sprint_17_vision_artificial.git
cd sprint_17_vision_artificial
```

2. **Crear un entorno virtual (recomendado):**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

## 📊 Preparación de Datos

### Formato del Dataset

El modelo espera los datos en el siguiente formato:

1. **Archivo de etiquetas** (`datasets/faces/labels.csv`):
```csv
file_name,real_age
imagen001.jpg,25
imagen002.jpg,17
imagen003.jpg,42
...
```

2. **Imágenes** en el directorio `datasets/faces/final_files/`:
   - Formato: JPG, PNG
   - Contenido: Fotografías de rostros
   - Organización: Todas en el mismo directorio

### Descarga del Dataset

Para obtener un dataset de rostros con edades, puedes utilizar:
- **UTKFace Dataset**: Conjunto de datos público con más de 20,000 imágenes
- **IMDB-WIKI Dataset**: Base de datos de celebridades con edades
- **ChaLearn Looking at People Dataset**: Dataset específico para estimación de edad

Asegúrate de procesar el dataset para que coincida con el formato esperado.

## 💻 Uso del Modelo

### Ejecutar el Notebook

1. **Iniciar Jupyter:**
```bash
jupyter notebook
```

2. **Abrir el notebook:** `age_estimation_model.ipynb`

3. **Ejecutar las celdas en orden:**
   - Importación de librerías
   - Carga y exploración de datos
   - Preprocesamiento
   - Creación del modelo
   - Entrenamiento
   - Evaluación

### Flujo del Modelo

1. **Carga de Datos**: Lee las imágenes y etiquetas del dataset
2. **Preprocesamiento**: 
   - Redimensiona imágenes a 224x224 píxeles
   - Normaliza valores de píxeles (0-1)
   - Aplica data augmentation en entrenamiento
3. **Modelo**: 
   - Arquitectura base: ResNet50 pre-entrenada
   - Transfer learning con capas personalizadas
   - Optimización en dos fases (congelado y fine-tuning)
4. **Entrenamiento**:
   - Fase 1: Solo capas superiores (base congelada)
   - Fase 2: Fine-tuning de últimas capas
5. **Evaluación**: 
   - Métricas: MAE, MSE, RMSE
   - Análisis específico para menores de edad
   - Visualizaciones de resultados

## 📈 Métricas de Rendimiento

El modelo utiliza las siguientes métricas:

- **MAE (Mean Absolute Error)**: Error promedio en años
- **MSE (Mean Squared Error)**: Error cuadrático medio
- **RMSE (Root Mean Squared Error)**: Raíz del error cuadrático medio
- **Recall**: Porcentaje de menores correctamente identificados (crítico para el caso de uso)
- **Precision**: De los clasificados como menores, cuántos realmente lo son

### Métricas Específicas para Menores de Edad

Para el caso de uso de prevención de venta de alcohol:
- Análisis de clasificación binaria (menor/mayor de 18 años)
- Matriz de confusión
- Énfasis en minimizar falsos negativos (menores clasificados como mayores)

## 🛠️ Tecnologías Utilizadas

- **TensorFlow/Keras**: Framework de deep learning
- **ResNet50**: Arquitectura de red neuronal convolucional
- **NumPy**: Computación numérica
- **Pandas**: Manipulación de datos
- **Matplotlib/Seaborn**: Visualización
- **Scikit-learn**: Métricas y utilidades de ML

## 📝 Resultados Esperados

El notebook genera:

1. **Visualizaciones**:
   - Distribución de edades en el dataset
   - Curvas de entrenamiento (loss y métricas)
   - Gráficos de dispersión (edad real vs predicha)
   - Distribución de errores
   - Análisis de residuos
   - MAE por rango de edad
   - Ejemplos de predicciones con imágenes

2. **Modelos Guardados**:
   - `best_age_model.h5`: Mejor modelo durante entrenamiento
   - `age_estimation_model_final.h5`: Modelo final en formato H5
   - `age_estimation_model_savedmodel/`: Modelo en formato SavedModel

3. **Métricas de Evaluación**:
   - Rendimiento general del modelo
   - Rendimiento específico para menores de edad
   - Análisis de falsos positivos y negativos

## ⚠️ Consideraciones Importantes

### Implementación en Producción

1. **Umbral de Decisión**: Se recomienda usar un margen de seguridad (ej: clasificar como menor si edad predicha < 21 años)
2. **Verificación Manual**: Implementar revisión humana para casos cerca del límite
3. **Recall > Precision**: Priorizar no dejar pasar menores (minimizar falsos negativos)

### Aspectos Éticos

1. **Privacidad**: Implementar políticas claras de manejo de imágenes
2. **Sesgos**: Verificar que el modelo no tenga sesgos por etnia, género, etc.
3. **Transparencia**: Mantener claridad sobre cómo se usan los datos
4. **Fallback**: Siempre tener opción de verificación manual

### Mejoras Futuras

- [ ] Aumentar el dataset con más ejemplos en el rango 16-20 años
- [ ] Explorar arquitecturas más modernas (EfficientNet, ViT)
- [ ] Implementar ensemble de modelos
- [ ] Agregar detección de calidad de imagen
- [ ] Considerar factores adicionales (iluminación, ángulo, etc.)

## 🤝 Contribuciones

Para contribuir al proyecto:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es parte del Sprint 17 de TripleTen.

## 📧 Contacto

Para preguntas o sugerencias sobre el proyecto, por favor abre un issue en GitHub.

---

**Nota**: Este modelo es una herramienta de apoyo y NO debe ser el único método de verificación de edad. Siempre debe complementarse con verificación manual y/o identificación oficial cuando sea necesario.