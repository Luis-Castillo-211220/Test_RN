import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

classes = ['anger', 'confused', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'shy', 'surprise']
img_rows, img_cols = 48,48
num_class = len(classes)

def load_data():
    data = []
    target = []
    
    for index, clase in enumerate(classes):
        folde_path = os.path.join('respaldo', clase)
        for img in os.listdir(folde_path):
            img_path = os.path.join(folde_path, img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image,(img_rows, img_cols))
            data.append(np.array(image))
            target.append(index)
    data = np.array(data)
    data = data.reshape(data.shape[0], img_rows, img_cols, 1)
    target = np.array(target)
    new_target = to_categorical(target, num_class)
    return data, new_target

data, target = load_data()

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape = (img_rows, img_cols, 1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0,25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_class, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=40, epochs=15, verbose=1, validation_data=(X_test, y_test))

model.save('modelo.h5')
# model.save('my_model.keras')

if not os.path.exists('graficas'):#si no existe la carpeta graficas
    os.makedirs('graficas')#crea la carpeta graficas

y_pred = model.predict(X_test)#hace una prediccion con los datos de prueba obteniendo las probabilidades de cada clase
y_pred_classes = np.argmax(y_pred,axis=1)#obtiene la clase con la probabilidad mas alta
y_true = np.argmax(y_test,axis=1)#obtiene la clase real
# que hace argmax? devuelve el indice del valor mas alto a lo largo de un eje
confusion_mtx = confusion_matrix(y_true, y_pred_classes)#crea una matriz de confusion


plt.figure(figsize=(8,6))#tamaño de la grafica
sns.heatmap(confusion_mtx, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)#grafica la matriz de confusion
plt.xlabel('Prediccion')#etiqueta del eje x
plt.ylabel('Real')#etiqueta del eje y
plt.savefig('graficas/matriz_confusion.png')#guarda la grafica
plt.show()#muestra la grafica

plt.figure()
plt.plot(history.history['loss'])#grafica el error del entrenamiento
plt.plot(history.history['val_loss'])#grafica el error de la validacion
plt.title('Historial de Error')#titulo
plt.ylabel('Error')#etiqueta del eje y
plt.xlabel('Época')#    etiqueta del eje x
plt.legend(['Entrenamiento', 'Validación'], loc='upper right')#leyenda
plt.savefig('graficas/historial_error.png')#guarda la grafica
plt.show()
