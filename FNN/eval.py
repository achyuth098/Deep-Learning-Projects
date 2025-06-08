import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fnn.utils import load_and_preprocess, load_model

def evaluate(args):
    (X_train, X_test, y_train, y_test), _ = load_and_preprocess(
        args.data_path, test_size=args.test_size
    )
    model = load_model(args.model_path)
    y_pred = (model.predict(X_test) > args.threshold).astype(np.int32)
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FNN model")
    parser.add_argument("--data_path",  type=str, default="fnn/data/Bank_Clents.csv")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_size",  type=float, default=0.2)
    parser.add_argument("--threshold",  type=float, default=0.5)
    args = parser.parse_args()
    evaluate(args)
