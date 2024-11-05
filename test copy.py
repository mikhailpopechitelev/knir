import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Пути к тренировочным и тестовым изображениям
train_dir = 'chest_xray/train'
val_dir = 'chest_xray/val'
test_dir = 'chest_xray/test'

# Полный путь к директории
if not os.path.exists(train_dir):
    print(f"Directory {train_dir} does not exist!")

# Генератор изображений с аугментацией
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=32, class_mode='binary')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(150, 150), batch_size=32, class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(150, 150), batch_size=32, class_mode='binary')

# Создание модели CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Для бинарной классификации
])

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

# Оценка модели на тестовых данных
test_loss, test_acc = model.evaluate(test_generator)
print(f'Точность на тестовых данных: {test_acc:.2f}')

# Визуализация обучения
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

model.save('pneumonia_model.h5')

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
model = load_model('pneumonia_model.h5')
# Функция для предсказания болезни на основе рентгеновского снимка
def predict_pneumonia(img_path):
    # Загрузить изображение и преобразовать его в нужный размер
    img = image.load_img(img_path, target_size=(150, 150))  # Размер должен совпадать с размером, который использовался при обучении
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Добавить размер батча
    img_array /= 255.0  # Нормализовать изображение
    
    # Сделать предсказание
    prediction = model.predict(img_array)
    # Вернуть результат
    if prediction[0] > 0.5:
        return 'Пневмония'
    else:
        return 'Норма'
    

result = predict_pneumonia('pnevmon.jpg') 
print(result)
