from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, classification_report
import tensorflow as tf

# Параметры данных
TIME_STEPS = 960  # Длина временного ряда
N_FEATURES = 52   # Количество признаков на каждом временном шаге
NUM_CLASSES = 11

def f1_metric(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)

    # Конвертация в one-hot для расчета по классам
    y_true = tf.one_hot(y_true, depth=21)
    y_pred = tf.one_hot(y_pred, depth=21)

    tp = tf.reduce_sum(y_true * y_pred, axis=0)  # True Positives per class
    fp = tf.reduce_sum(y_pred, axis=0) - tp  # False Positives
    fn = tf.reduce_sum(y_true, axis=0) - tp  # False Negatives

    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), 0.0, f1)  # Замена NaN на 0
    return tf.reduce_mean(f1)  # Macro F1

print("Считывание данных")
df = pd.read_csv("../datasets/trucated_data/test_faultNumbers_1-2-3-4-5-6-7-8-9-10.csv")
print(df['faultNumber'].unique())
df.dropna()
X = df.drop('faultNumber', axis=1).drop('simulationRun', axis=1).drop('sample', axis=1)
y = df['faultNumber']

y_compressed = []
for i in range(len(y) // TIME_STEPS):
    y_compressed.append(y[i*TIME_STEPS])

y_compressed = np.array(y_compressed)
scaler = StandardScaler()
X = scaler.fit_transform(X.values.reshape(-1, N_FEATURES)).reshape(-1, TIME_STEPS, N_FEATURES)
X_test = X
y_test = y_compressed
print(y_test)
model = load_model('tep_classifier_10.keras')

# Получение предсказаний модели (вероятности для каждого класса)
y_pred_probs = model.predict(X_test)
# Преобразование вероятностей в метки классов
print(y_pred_probs)
y_pred = np.argmax(y_pred_probs, axis=1)
f1 = f1_score(y_test, y_pred, average="weighted")
print(f"F1-score: {f1:.4f}")

# Отчет по классам
print(classification_report(y_test, y_pred, average='weighted'))

# cm = confusion_matrix(y_test, y_pred)
# print(cm)
#
# plt.figure(figsize=(12, 8))
# sns.heatmap(
#     cm,
#     annot=True,
#     cmap="Blues",
#     xticklabels=range(0,NUM_CLASSES),
#     yticklabels=range(0,NUM_CLASSES),
#     square=True
# )
# plt.show()


# test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
# print(f'\nTest Accuracy: {test_acc:.4f}')
# print(f'Test Loss: {test_loss:.4f}')