import numpy as np
import pandas as pd
import os


# Генерация данных
def generate_data(num_points, noise_level=0.1, anomalies=False):
    # Создаем папки для данных, если они не существуют
    os.makedirs('train', exist_ok=True)
    os.makedirs('test', exist_ok=True)

    x = np.linspace(0, 10, num_points)
    y = np.sin(x) + np.random.normal(0, noise_level, num_points)
    if anomalies:
        y[np.random.choice(num_points, 5)] += np.random.uniform(-5, 5, 5)
    return pd.DataFrame({'x': x, 'y': y})


# Генерация тренировочных и тестовых данных
train_data = generate_data(100, anomalies=True)
test_data = generate_data(50)

# Сохранение данных
train_data.to_csv('train/train_data.csv', index=False)
test_data.to_csv('test/test_data.csv', index=False)
