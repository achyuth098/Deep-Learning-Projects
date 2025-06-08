import argparse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from cnn.utils import get_data_generators, save_model

def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train(args):
    train_gen, val_gen = get_data_generators(
        args.data_dir,
        img_size=(args.img_height, args.img_width),
        batch_size=args.batch_size,
        validation_split=args.validation_split
    )

    model = build_cnn(
        input_shape=(args.img_height, args.img_width, 3),
        num_classes=train_gen.num_classes
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(args.checkpoint_path, save_best_only=True, monitor='val_accuracy')
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2
    )

    save_model(model, args.output_path)
    print(f"Model saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a simple CNN")
    parser.add_argument("--data_dir",           type=str,   required=True)
    parser.add_argument("--img_height",         type=int,   default=224)
    parser.add_argument("--img_width",          type=int,   default=224)
    parser.add_argument("--batch_size",         type=int,   default=32)
    parser.add_argument("--validation_split",   type=float, default=0.2)
    parser.add_argument("--learning_rate",      type=float, default=1e-3)
    parser.add_argument("--epochs",             type=int,   default=20)
    parser.add_argument("--checkpoint_path",    type=str,   default="cnn/models/best.h5")
    parser.add_argument("--output_path",        type=str,   default="cnn/models/final_model")
    args = parser.parse_args()
    train(args)
