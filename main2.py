def predict_fraud(transaction_amount, transaction_time):
    transaction = scaler.transform([[transaction_amount, transaction_time]])
    prediction = model.predict(transaction)
    return "Fraudulent" if prediction[0] == 1 else "Non-Fraudulent"

print(predict_fraud(7000, 25))
print(predict_fraud(200, 45))
