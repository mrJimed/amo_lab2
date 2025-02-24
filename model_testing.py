import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

# Загрузка модели
model = joblib.load('model.pkl')

# Загрузка тестовых данных
test_data = pd.read_csv('test/test_data_scaled.csv')

# Предсказание и оценка модели
predictions = model.predict(test_data[['x']])
mse = mean_squared_error(test_data['y'], predictions)
print(f'Mean Squared Error: {mse}')