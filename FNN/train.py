import argparse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from fnn.utils import load_and_preprocess, save_model

def build_model(input_dim):
    model = Sequential([
        Dense(32, activation='tanh', input_shape=(input_dim,), kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='tanh', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='tanh', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model

def train(args):
    (X_train, X_test, y_train, y_test), _ = load_and_preprocess(
        args.data_path, test_size=args.test_size
    )
    model = build_model(X_train.shape[1])
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    early = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=None,
        callbacks=[early],
        verbose=2
    )
    save_model(model, args.output_path)
    print(f"Model saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FNN for churn prediction")
    parser.add_argument("--data_path",    type=str,   required=True)
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=5e-4)
    parser.add_argument("--test_size",    type=float, default=0.2)
    parser.add_argument("--output_path",  type=str,   default="fnn/models/churn_model")
    args = parser.parse_args()
    train(args)
