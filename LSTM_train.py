import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import keras

# Параметры данных
TIME_STEPS = 900*5  # Длина временного ряда
N_FEATURES = 15   # Количество признаков на каждом временном шаге
NUM_CLASSES = 3

print("Считывание данных")
import os
def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r

#df = pd.read_csv("../datasets/trucated_data/train_faultNumbers_1-2-3-4-5-6-7-8-9-10.csv")
data_files = list_files("data/14.04 (15 минут, 5 точек в секунду, поворот в первой половине сценария)/")
df = pd.concat((pd.read_parquet(f) for f in data_files))

df.dropna()
X = df.drop('fault_type', axis=1).drop('fault_active', axis=1).drop('fault_time', axis=1).drop('sim_id', axis=1)
y = df['fault_type']
X.info()

scaler = StandardScaler()
X = scaler.fit_transform(X.values.reshape(-1, N_FEATURES)).reshape(-1, TIME_STEPS, N_FEATURES)

y_compressed = []
for i in range(len(y) // TIME_STEPS):
    # try:
    y_compressed.append(y.iloc[i*TIME_STEPS])
    # except Exception as e:


print(pd.unique(y_compressed))
y_compressed = np.array(y_compressed)
print(np.unique(y_compressed, axis=0))


X_train = X
y_train = y_compressed
y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
print(y_train)
X_train, y_train = shuffle(X_train, y_train, random_state=42)

model = Sequential()
# Входной слой
model.add(LSTM(units=64,
               input_shape=(None, N_FEATURES),
               return_sequences=True))
model.add(Dropout(0.2))
# Второй LSTM слой
# model.add(LSTM(40,
#                return_sequences=True))
# model.add(Dropout(0.2))
# Второй LSTM слой
model.add(LSTM(25,
               return_sequences=True))
model.add(Dropout(0.2))
# Третий LSTM слой
model.add(LSTM(12,
               return_sequences=False))
model.add(Dropout(0.2))
# Выходные слой
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            # loss='binary_crossentropy',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
# Вывод структуры модели
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    epochs=20,
                    batch_size=50,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=1)

model.save("models/tep_classifier_10.keras")

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()
plt.show()