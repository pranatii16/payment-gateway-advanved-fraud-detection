import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = {
    'transaction_amount': [100, 2000, 5000, 7000, 10000, 15000],
    'transaction_time': [10, 50, 30, 20, 60, 40],
    'is_fraud': [0, 0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[['transaction_amount', 'transaction_time']]
y = df['is_fraud']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
