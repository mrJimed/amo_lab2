import pandas as pd
from sklearn.preprocessing import StandardScaler

# Загрузка данных
train_data = pd.read_csv('train/train_data.csv')
test_data = pd.read_csv('test/test_data.csv')

# Предобработка данных
scaler = StandardScaler()
train_data[['x', 'y']] = scaler.fit_transform(train_data[['x', 'y']])
test_data[['x', 'y']] = scaler.transform(test_data[['x', 'y']])

# Сохранение предобработанных данных
train_data.to_csv('train/train_data_scaled.csv', index=False)
test_data.to_csv('test/test_data_scaled.csv', index=False)