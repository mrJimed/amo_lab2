.\.venv\Scripts\activate.bat

pip install numpy pandas scikit-learn joblib

# Генерация данных
python data_generation.py

# Предобработка данных
python model_preprocessing.py

# Обучение модели
python model_training.py

# Тестирование модели
python model_testing.py

.\.venv\Scripts\deactivate.bat