# Guía de Estructura de Datos

## Formato del Dataset

Para usar el modelo de estimación de edad, necesitas organizar tus datos de la siguiente manera:

### Estructura de Directorios

```
datasets/
└── faces/
    ├── labels.csv
    └── final_files/
        ├── 001.jpg
        ├── 002.jpg
        ├── 003.jpg
        └── ...
```

### Archivo de Etiquetas (labels.csv)

El archivo CSV debe tener exactamente estas columnas:

| file_name | real_age |
|-----------|----------|
| 001.jpg   | 25       |
| 002.jpg   | 17       |
| 003.jpg   | 42       |
| 004.jpg   | 19       |

**Especificaciones:**
- **file_name**: Nombre del archivo de imagen (incluyendo extensión)
- **real_age**: Edad real de la persona en la imagen (número entero)

**Ejemplo de archivo CSV:**
```csv
file_name,real_age
001.jpg,25
002.jpg,17
003.jpg,42
004.jpg,19
005.jpg,33
```

### Imágenes

**Ubicación:** `datasets/faces/final_files/`

**Formatos aceptados:**
- JPG/JPEG
- PNG

**Recomendaciones:**
- Imágenes de rostros centrados
- Buena iluminación
- Resolución mínima: 224x224 píxeles
- Evitar imágenes borrosas o con oclusiones

## Datasets Públicos Recomendados

### 1. UTKFace Dataset
- **Descripción:** Más de 20,000 imágenes de rostros con edades de 0 a 116 años
- **URL:** https://susanqq.github.io/UTKFace/
- **Formato:** Imágenes JPG con edad, género y etnia codificados en el nombre
- **Procesamiento necesario:** Extraer edad del nombre del archivo y crear labels.csv

### 2. IMDB-WIKI Dataset
- **Descripción:** ~500,000 imágenes de celebridades con metadatos
- **URL:** https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
- **Formato:** Archivos .mat con metadatos
- **Procesamiento necesario:** Extraer información de archivos .mat

### 3. AgeDB Dataset
- **Descripción:** ~16,000 imágenes con edades verificadas manualmente
- **URL:** https://ibug.doc.ic.ac.uk/resources/agedb/
- **Formato:** Imágenes con anotaciones en archivo de texto
- **Procesamiento necesario:** Convertir anotaciones a formato CSV

## Script de Ejemplo para Procesar UTKFace

```python
import pandas as pd
from pathlib import Path
import shutil

def process_utkface_dataset(source_dir, target_dir):
    """
    Procesa el dataset UTKFace al formato esperado.
    
    Formato de nombre UTKFace: [age]_[gender]_[race]_[date&time].jpg
    Ejemplo: 25_1_0_20170116174525125.jpg
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    final_files = target_path / 'final_files'
    
    # Crear directorios
    final_files.mkdir(parents=True, exist_ok=True)
    
    # Procesar imágenes
    labels = []
    counter = 1
    
    for img_file in source_path.glob('*.jpg'):
        try:
            # Extraer edad del nombre
            age = int(img_file.name.split('_')[0])
            
            # Nuevo nombre
            new_name = f'{counter:06d}.jpg'
            
            # Copiar imagen
            shutil.copy(img_file, final_files / new_name)
            
            # Agregar a etiquetas
            labels.append({'file_name': new_name, 'real_age': age})
            counter += 1
            
        except (ValueError, IndexError):
            print(f'Advertencia: No se pudo procesar {img_file.name}')
            continue
    
    # Crear CSV
    df = pd.DataFrame(labels)
    df.to_csv(target_path / 'labels.csv', index=False)
    
    print(f'✓ Procesadas {len(labels)} imágenes')
    print(f'✓ Dataset guardado en {target_path}')

# Uso
# process_utkface_dataset('path/to/UTKFace', 'datasets/faces')
```

## Validación del Dataset

Antes de entrenar, valida tu dataset con este script:

```python
import pandas as pd
from pathlib import Path

def validate_dataset(data_dir):
    """Valida que el dataset esté correctamente formateado."""
    
    data_path = Path(data_dir)
    labels_file = data_path / 'labels.csv'
    images_dir = data_path / 'final_files'
    
    print("Validando dataset...")
    
    # Verificar estructura
    if not labels_file.exists():
        print("❌ Error: No se encontró labels.csv")
        return False
    
    if not images_dir.exists():
        print("❌ Error: No se encontró el directorio final_files")
        return False
    
    # Cargar etiquetas
    df = pd.read_csv(labels_file)
    
    # Verificar columnas
    if 'file_name' not in df.columns or 'real_age' not in df.columns:
        print("❌ Error: El CSV debe tener columnas 'file_name' y 'real_age'")
        return False
    
    # Verificar que las imágenes existan
    missing_images = []
    for file_name in df['file_name']:
        img_path = images_dir / file_name
        if not img_path.exists():
            missing_images.append(file_name)
    
    if missing_images:
        print(f"❌ Error: {len(missing_images)} imágenes no encontradas")
        print(f"   Primeras faltantes: {missing_images[:5]}")
        return False
    
    # Estadísticas
    print(f"\n✓ Dataset válido!")
    print(f"  Total de imágenes: {len(df)}")
    print(f"  Rango de edades: {df['real_age'].min()} - {df['real_age'].max()}")
    print(f"  Edad promedio: {df['real_age'].mean():.1f}")
    
    return True

# Uso
# validate_dataset('datasets/faces')
```

## Notas Importantes

1. **Tamaño del Dataset:** Se recomienda tener al menos 1,000 imágenes para entrenamiento
2. **Distribución de Edades:** Intenta tener una distribución balanceada de edades
3. **Calidad de Imágenes:** La calidad del modelo depende directamente de la calidad de las imágenes
4. **Privacidad:** Asegúrate de tener los permisos necesarios para usar las imágenes
5. **Ética:** Verifica que el dataset no tenga sesgos significativos por raza, género, etc.
