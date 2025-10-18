"""
Script de ejemplo para usar el modelo de estimación de edad.

Este script demuestra cómo cargar un modelo entrenado y hacer
predicciones sobre nuevas imágenes.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from tensorflow import keras
from utils import predict_age, preprocess_single_image


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description='Predice la edad de una persona a partir de una imagen'
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Ruta a la imagen para analizar'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='best_age_model.h5',
        help='Ruta al modelo entrenado (por defecto: best_age_model.h5)'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=18,
        help='Umbral de edad para clasificación (por defecto: 18)'
    )
    
    args = parser.parse_args()
    
    # Verificar que la imagen existe
    img_path = Path(args.image_path)
    if not img_path.exists():
        print(f"Error: No se encontró la imagen en {args.image_path}")
        sys.exit(1)
    
    # Verificar que el modelo existe
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: No se encontró el modelo en {args.model}")
        print("Por favor, entrena el modelo primero usando el notebook.")
        sys.exit(1)
    
    print("=" * 60)
    print("SISTEMA DE ESTIMACIÓN DE EDAD - GOOD SEED")
    print("=" * 60)
    print(f"\n📷 Imagen: {img_path.name}")
    print(f"🤖 Modelo: {model_path.name}")
    print(f"⚠️  Umbral de edad: {args.threshold} años\n")
    
    try:
        # Cargar modelo
        print("Cargando modelo...")
        model = keras.models.load_model(args.model)
        print("✓ Modelo cargado exitosamente\n")
        
        # Hacer predicción
        print("Analizando imagen...")
        predicted_age = predict_age(model, str(img_path))
        print("✓ Análisis completado\n")
        
        # Mostrar resultados
        print("=" * 60)
        print("RESULTADOS")
        print("=" * 60)
        print(f"\n🎂 Edad estimada: {predicted_age:.1f} años")
        
        # Clasificación
        is_minor = predicted_age < args.threshold
        if is_minor:
            print(f"⚠️  ATENCIÓN: Persona clasificada como MENOR de {args.threshold} años")
            print("   → Se requiere verificación de identificación oficial")
        else:
            print(f"✓ Persona clasificada como MAYOR de {args.threshold} años")
            print("   → Aún así, se recomienda verificación de identificación")
        
        # Margen de seguridad
        margin = abs(predicted_age - args.threshold)
        if margin < 3:
            print(f"\n⚠️  ADVERTENCIA: Edad muy cerca del umbral (margen: {margin:.1f} años)")
            print("   → Se recomienda VERIFICACIÓN MANUAL OBLIGATORIA")
        
        print("\n" + "=" * 60)
        print("IMPORTANTE: Este sistema es una HERRAMIENTA DE APOYO.")
        print("SIEMPRE verifique con identificación oficial cuando sea necesario.")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error al procesar la imagen: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
