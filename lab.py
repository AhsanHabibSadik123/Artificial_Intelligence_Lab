import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#x_train = x_train.astype("float32") / 255.0
#x_test = x_test.astype("float32") / 255.0

#x_train = x_train[..., None]
#x_test = x_test[..., None]

for i in range(len(y_train)):
    if y_train[i] % 2 == 1:
        y_train[i] = 0
    else:
        y_train[i] = 1

for i in range(len(y_test)):
    if y_test[i] % 2 == 1:
        y_test[i] = 0
    else:
        y_test[i] = 1

x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size = .15,
    stratify=y_train
)
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(16, (3, 3), activation='relu'),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(.35),
    Dense(64, activation='relu'),
    Dropout(.2),
    Dense(2, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    jit_compile=False
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience = 3
)

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks= [early_stopping]
)

result = model.evaluate(x_test, y_test)
print(result)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

print("OK")