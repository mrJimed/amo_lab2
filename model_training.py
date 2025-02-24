import pandas as pd
from sklearn.linear_model import LinearRegression

# Загрузка предобработанных данных
train_data = pd.read_csv('train/train_data_scaled.csv')

# Обучение модели
model = LinearRegression()
model.fit(train_data[['x']], train_data['y'])

# Сохранение модели
import joblib
joblib.dump(model, 'model.pkl')