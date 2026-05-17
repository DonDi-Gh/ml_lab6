import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# 1. Загружаем обученную модель
model = tf.keras.models.load_model("gesture_model.h5")

# 2. Загружаем и готовим ваше фото
img_path = "img_1062.jpg"  # Убедитесь, что файл с таким именем есть в папке
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 3. Делаем предсказание
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

print(f"--- РЕЗУЛЬТАТ ---")
print(f"Нейросеть считает, что на фото цифра: {predicted_class}")
