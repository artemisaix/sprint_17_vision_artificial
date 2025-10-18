"""
Utilidades para el modelo de estimación de edad.

Este módulo contiene funciones auxiliares para cargar datos,
preprocesar imágenes y evaluar el modelo.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_labels(labels_file: str) -> Optional[pd.DataFrame]:
    """
    Carga el archivo de etiquetas y valida su formato.
    
    Args:
        labels_file: Ruta al archivo CSV con las etiquetas
    
    Returns:
        DataFrame con las etiquetas o None si hay error
    """
    labels_path = Path(labels_file)
    
    if not labels_path.exists():
        print(f"Error: No se encontró el archivo {labels_file}")
        return None
    
    try:
        df = pd.read_csv(labels_file)
        
        # Validar columnas requeridas
        required_cols = ['file_name', 'real_age']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: El archivo debe contener las columnas {required_cols}")
            return None
        
        print(f"✓ Etiquetas cargadas correctamente: {len(df)} registros")
        return df
        
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None


def create_data_generators(
    df: pd.DataFrame,
    img_dir: str,
    img_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = 42
) -> Tuple:
    """
    Crea generadores de datos para entrenamiento, validación y prueba.
    
    Args:
        df: DataFrame con file_name y real_age
        img_dir: Directorio con las imágenes
        img_size: Tamaño de las imágenes (por defecto 224 para ResNet)
        batch_size: Tamaño del batch
        val_split: Proporción para validación
        test_split: Proporción para prueba
        seed: Semilla para reproducibilidad
    
    Returns:
        Tupla con (train_gen, val_gen, test_gen, train_df, val_df, test_df)
    """
    # Dividir datos
    train_val_df, test_df = train_test_split(
        df, test_size=test_split, random_state=seed
    )
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_split/(1-test_split), 
        random_state=seed
    )
    
    print(f"Datos divididos:")
    print(f"  Entrenamiento: {len(train_df)} imágenes")
    print(f"  Validación: {len(val_df)} imágenes")
    print(f"  Prueba: {len(test_df)} imágenes")
    
    # Generador de datos con aumento para entrenamiento
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    # Generador para validación y prueba (solo reescalado)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Crear generadores
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=img_dir,
        x_col='file_name',
        y_col='real_age',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='raw',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=img_dir,
        x_col='file_name',
        y_col='real_age',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='raw',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=img_dir,
        x_col='file_name',
        y_col='real_age',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='raw',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator, train_df, val_df, test_df


def evaluate_age_classification(
    true_ages: np.ndarray,
    predicted_ages: np.ndarray,
    threshold: int = 18
) -> dict:
    """
    Evalúa el modelo como clasificador binario (menor/mayor de edad).
    
    Args:
        true_ages: Edades reales
        predicted_ages: Edades predichas
        threshold: Umbral de edad (por defecto 18)
    
    Returns:
        Diccionario con métricas de clasificación
    """
    true_minors = true_ages < threshold
    pred_minors = predicted_ages < threshold
    
    # Matriz de confusión
    tp = np.sum(true_minors & pred_minors)  # Verdaderos positivos
    tn = np.sum(~true_minors & ~pred_minors)  # Verdaderos negativos
    fp = np.sum(~true_minors & pred_minors)  # Falsos positivos
    fn = np.sum(true_minors & ~pred_minors)  # Falsos negativos
    
    # Métricas
    accuracy = (tp + tn) / len(true_ages) if len(true_ages) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def calculate_regression_metrics(
    true_ages: np.ndarray,
    predicted_ages: np.ndarray
) -> dict:
    """
    Calcula métricas de regresión para el modelo.
    
    Args:
        true_ages: Edades reales
        predicted_ages: Edades predichas
    
    Returns:
        Diccionario con métricas de regresión
    """
    mae = mean_absolute_error(true_ages, predicted_ages)
    mse = mean_squared_error(true_ages, predicted_ages)
    rmse = np.sqrt(mse)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }


def analyze_by_age_range(
    true_ages: np.ndarray,
    predicted_ages: np.ndarray,
    ranges: list = None
) -> pd.DataFrame:
    """
    Analiza el rendimiento del modelo por rangos de edad.
    
    Args:
        true_ages: Edades reales
        predicted_ages: Edades predichas
        ranges: Lista de tuplas (inicio, fin) para los rangos
    
    Returns:
        DataFrame con MAE por rango de edad
    """
    if ranges is None:
        ranges = [(0, 18), (18, 25), (25, 35), (35, 50), (50, 100)]
    
    results = []
    
    for start, end in ranges:
        mask = (true_ages >= start) & (true_ages < end)
        if mask.sum() > 0:
            range_mae = mean_absolute_error(true_ages[mask], predicted_ages[mask])
            results.append({
                'rango': f'{start}-{end}',
                'n_muestras': mask.sum(),
                'mae': range_mae
            })
    
    return pd.DataFrame(results)


def preprocess_single_image(
    img_path: str,
    img_size: int = 224
) -> np.ndarray:
    """
    Preprocesa una imagen individual para predicción.
    
    Args:
        img_path: Ruta a la imagen
        img_size: Tamaño objetivo
    
    Returns:
        Array de numpy con la imagen preprocesada
    """
    from tensorflow.keras.preprocessing import image
    
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    return img_array


def predict_age(
    model,
    img_path: str,
    img_size: int = 224
) -> float:
    """
    Predice la edad de una persona en una imagen.
    
    Args:
        model: Modelo entrenado de Keras
        img_path: Ruta a la imagen
        img_size: Tamaño de la imagen
    
    Returns:
        Edad predicha
    """
    img_array = preprocess_single_image(img_path, img_size)
    prediction = model.predict(img_array, verbose=0)
    return float(prediction[0][0])


def print_evaluation_report(
    regression_metrics: dict,
    classification_metrics: dict,
    age_range_analysis: pd.DataFrame = None
):
    """
    Imprime un reporte completo de evaluación.
    
    Args:
        regression_metrics: Métricas de regresión
        classification_metrics: Métricas de clasificación
        age_range_analysis: Análisis por rango de edad
    """
    print("=" * 70)
    print("REPORTE DE EVALUACIÓN DEL MODELO")
    print("=" * 70)
    
    print("\n📊 MÉTRICAS DE REGRESIÓN:")
    print(f"  MAE (Error Absoluto Medio): {regression_metrics['mae']:.2f} años")
    print(f"  MSE (Error Cuadrático Medio): {regression_metrics['mse']:.2f}")
    print(f"  RMSE (Raíz del ECM): {regression_metrics['rmse']:.2f} años")
    
    print("\n🎯 CLASIFICACIÓN BINARIA (Menor/Mayor de 18 años):")
    print(f"  Verdaderos Positivos: {classification_metrics['true_positives']}")
    print(f"  Verdaderos Negativos: {classification_metrics['true_negatives']}")
    print(f"  Falsos Positivos: {classification_metrics['false_positives']}")
    print(f"  Falsos Negativos: {classification_metrics['false_negatives']}")
    print(f"\n  Accuracy: {classification_metrics['accuracy']:.2%}")
    print(f"  Precision: {classification_metrics['precision']:.2%}")
    print(f"  Recall: {classification_metrics['recall']:.2%}")
    print(f"  F1-Score: {classification_metrics['f1_score']:.2%}")
    
    if age_range_analysis is not None:
        print("\n📈 RENDIMIENTO POR RANGO DE EDAD:")
        print(age_range_analysis.to_string(index=False))
    
    print("\n" + "=" * 70)
