import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from cnn.utils import get_data_generators, load_model

def evaluate(args):
    _, val_gen = get_data_generators(
        args.data_dir,
        img_size=(args.img_height, args.img_width),
        batch_size=args.batch_size,
        validation_split=args.validation_split
    )

    model = load_model(args.model_path)
    val_gen.reset()

    preds = model.predict(val_gen, verbose=0)
    y_true = val_gen.classes
    y_pred = np.argmax(preds, axis=1)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys())))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate CNN model")
    parser.add_argument("--data_dir",         type=str, required=True)
    parser.add_argument("--img_height",       type=int, default=224)
    parser.add_argument("--img_width",        type=int, default=224)
    parser.add_argument("--batch_size",       type=int, default=32)
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--model_path",       type=str, required=True)
    args = parser.parse_args()
    evaluate(args)
