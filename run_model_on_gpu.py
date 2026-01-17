
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam


def load_train(path):
    
    """
    Carga la parte de entrenamiento del conjunto de datos desde la ruta.
    """
    
    # coloca tu código aquí
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.25,   # 75% train, 25% valid
        horizontal_flip=True,
        rotation_range=20,
        zoom_range=0.2
    )
    
    train_gen_flow = datagen.flow_from_dataframe(
        dataframe=pd.read_csv('/datasets/faces/labels.csv'),
        directory=path,
        x_col='file_name',
        y_col='real_age',
        target_size=(224,224),
        batch_size=32,
        class_mode='raw',
        subset='training',
        seed=42
    )
    return train_gen_flow

    return train_gen_flow


def load_test(path):
    
    """
    Carga la parte de validación/prueba del conjunto de datos desde la ruta
    """
    
    #  coloca tu código aquí
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.25
    )
    
    test_gen_flow = datagen.flow_from_dataframe(
        dataframe=pd.read_csv('/datasets/faces/labels.csv'),
        directory=path,
        x_col='file_name',
        y_col='real_age',
        target_size=(224,224),
        batch_size=32,
        class_mode='raw',
        subset='validation',
        seed=42
    )


    return test_gen_flow


def create_model(input_shape):
    
    """
    Define el modelo
    """
    
    #  coloca tu código aquí
    backbone = ResNet50(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False
    )
    backbone.trainable = False  # congelamos las capas base
    
    model = Sequential([
        backbone,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='linear')  # regresión: salida numérica
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='mean_absolute_error',
        metrics=['mae']
    )


    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=20,
                steps_per_epoch=None, validation_steps=None):

    """
    Entrena el modelo dados los parámetros
    """
    
    #  coloca tu código aquí
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2
    )


    return model




if __name__ == "__main__":
    
    # 1. Definir la ruta de los datos en la plataforma GPU
    # OJO: La ruta en la plataforma es diferente a la de tu notebook
    PATH = '/datasets/faces/'
    
    # 2. Cargar los generadores de datos
    print("Cargando datos de entrenamiento...")
    train_data = load_train(PATH + 'final_files/')
    
    print("Cargando datos de validación...")
    test_data = load_test(PATH + 'final_files/')
    
    # 3. Crear el modelo
    print("Creando modelo...")
    model = create_model(input_shape=(224, 224, 3))
    
    # 4. Entrenar el modelo
    print("Iniciando entrenamiento...")
    model = train_model(
        model=model,
        train_data=train_data,
        test_data=test_data,
        epochs=20,  # Puedes ajustar esto, 20 es un buen inicio
        # Los 'steps' se infieren automáticamente de los generadores
        steps_per_epoch=len(train_data), 
        validation_steps=len(test_data)
    )
    
    print("Entrenamiento finalizado.")

