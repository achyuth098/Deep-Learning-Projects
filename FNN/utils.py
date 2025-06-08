import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def load_and_preprocess(data_path, test_size=0.2, random_state=42):
    df = pd.read_csv(data_path)
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"], errors='ignore')
    X = pd.get_dummies(df.drop(columns=["Exited"]), columns=["Geography", "Gender"], drop_first=True)
    y = df["Exited"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return (X_train, X_test, y_train, y_test), scaler

def save_model(model, path):
    model.save(path)

def load_model(path):
    return tf.keras.models.load_model(path)
