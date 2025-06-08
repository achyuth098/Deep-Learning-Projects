import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def main(args):
    df = pd.read_csv(args.predictions_csv)

    y_true = df['Sentiment']
    for model in ('TextBlob_Pred','VADER_Pred'):
        print(f"\n=== Evaluation: {model} ===")
        print(classification_report(y_true, df[model]))
        cm = confusion_matrix(y_true, df[model], labels=['Positive','Negative'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=['Positive','Negative'])
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f"Confusion Matrix — {model}")
        plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser("Evaluate sentiment-predictions")
    p.add_argument("--predictions_csv", required=True,
                   help="CSV produced by train.py, with Sentiment and *_Pred columns")
    args = p.parse_args()
    main(args)
