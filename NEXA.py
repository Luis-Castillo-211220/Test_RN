import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

# Configuraciones iniciales
img_width, img_height = 48, 48
train_data_dir = "./imagenes/Entrenamiento"
validation_data_dir = "./imagenes/Validacion"
num_classes = 8  # Ajusta esto según el número de clases que tengas|
batch_size = 32
epochss = 10

# Preparación de los generadores de datos
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    color_mode="grayscale",
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    color_mode="grayscale",
)

# Construcción del modelo
model = Sequential(
    [
        Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(img_width, img_height, 1),
        ),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Resumen del modelo
model.summary()

# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenamiento del modelo
history = model.fit(
    train_generator,
    # steps_per_epoch=train_generator.samples // batch_size,
    steps_per_epoch = train_generator.samples // train_generator.batch_size,
    epochs = epochss,
    validation_data=validation_generator,
    # validation_steps=validation_generator.samples // batch_size,
    validation_steps = validation_generator.samples // validation_generator.batch_size
    # callbacks=[early_stopping]
)

# Guardar el modelo
model.save("modelo_emociones.h5")

# Opcional: Gráficas de precisión y pérdida
import matplotlib.pyplot as plt

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Precisión de Entrenamiento")
plt.plot(epochs_range, val_acc, label="Precisión de Validación")
plt.legend(loc="lower right")
plt.title("Precisión de Entrenamiento y Validación")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Pérdida de Entrenamiento")
plt.plot(epochs_range, val_loss, label="Pérdida de Validación")
plt.legend(loc="upper right")
plt.title("Pérdida de Entrenamiento y Validación")
plt.show()
