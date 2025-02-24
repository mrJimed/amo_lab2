#!/bin/bash

# Установка зависимостей
echo "Установка зависимостей..."
pip install numpy pandas scikit-learn joblib

# Генерация данных
echo "Генерация данных..."
python data_generation.py

# Предобработка данных
echo "Предобработка данных..."
python model_preprocessing.py

# Обучение модели
echo "Обучение модели..."
python model_training.py

# Тестирование модели
echo "Тестирование модели..."
python model_testing.py

echo "Конвейер завершен."