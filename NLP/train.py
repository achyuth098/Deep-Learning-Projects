import os
import argparse
from NLP.utils import load_data, preprocess_text, textblob_sentiment, vader_sentiment

def main(args):
    df = load_data(args.data_path)

    # preprocess
    df['Cleaned'] = df['Review'].apply(preprocess_text)

    # predict with two methods
    df['TextBlob_Pred'] = df['Cleaned'].apply(textblob_sentiment)
    df['VADER_Pred']    = df['Cleaned'].apply(vader_sentiment)

    # save to CSV
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df.to_csv(args.output_path, index=False)
    print(f"Predictions saved to {args.output_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser("Compute sentiments with TextBlob & VADER")
    p.add_argument("--data_path",   required=True,
                   help="CSV of reviews with columns Review,Rating")
    p.add_argument("--output_path", default="NLP/data/predictions.csv",
                   help="Where to write Review,Sentiment,TextBlob_Pred,VADER_Pred")
    args = p.parse_args()
    main(args)
