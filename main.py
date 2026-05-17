import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# 1. Подготовка данных
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 20% на валидацию
)


DATASET_PATH = 'dataset/' 

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 2. Создание модели CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    # Количество выходных нейронов равно количеству папок-классов
    tf.keras.layers.Dense(len(train_data.class_indices), activation='softmax')
])

# 3. Компиляция и обучение
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

print("Начало обучения...")
model.fit(train_data, validation_data=val_data, epochs=10)

# 4. Оценка модели
loss, accuracy = model.evaluate(val_data)
print(f'Точность модели на валидации: {accuracy * 100:.2f}%')

# Сохраним модель, чтобы не переобучать каждый раз
model.save('gesture_model.h5')