# Estimación de Edad a partir de Imágenes de Rostros

## Descripción del Proyecto
Sistema de visión por computadora que estima la edad de personas a partir de imágenes de rostros utilizando deep learning. El modelo está basado en ResNet50 con fine-tuning para regresión.

## Habilidades Técnicas

### Deep Learning & Visión por Computadora
- **Arquitectura**: ResNet50 con capas personalizadas
- **Tipo de Problema**: Regresión para estimación de edad
- **Aumento de Datos**: Flip horizontal, rotación (±20°), zoom (20%)
- **Preprocesamiento**: Normalización de píxeles (0-1)

### Stack Tecnológico
- **Lenguaje**: Python 3.8+
- **ML Framework**: TensorFlow/Keras
- **Procesamiento de Datos**: Pandas, NumPy
- **Manejo de Imágenes**: OpenCV (implícito en el flujo)

## Estructura del Código
```python
run_model_on_gpu.py
├── load_train()       # Carga y aumenta datos de entrenamiento
├── load_test()        # Carga datos de validación
├── create_model()     # Define la arquitectura del modelo
└── train_model()      # Entrena el modelo
```

## Configuración del Entorno
```bash
conda create -n age_estimation python=3.8
conda activate age_estimation
pip install tensorflow pandas
```

## Uso
1. Organiza tus datos:
   ```
   /ruta/a/tus/datos/
   ├── train/      # Imágenes de entrenamiento
   ├── val/        # Imágenes de validación
   └── labels.csv  # Edades correspondientes
   ```

2. Ejecuta el entrenamiento:
   ```bash
   python run_model_on_gpu.py
   ```

## Características Clave
- **Eficiente**: Usa fine-tuning en ResNet50
- **Robusto**: Aumento de datos integrado
- **Métricas**: Monitorea MAE durante el entrenamiento

## Mejoras Futuras
- [ ] Implementar inferencia en tiempo real
- [ ] Añadir soporte para video
- [ ] Crear API REST

## Contacto
[Andrea Carballo]  
[[LinkedIn](https://www.linkedin.com/in/andreacarballo/)]  
[diana.carballo@live.com.mx]