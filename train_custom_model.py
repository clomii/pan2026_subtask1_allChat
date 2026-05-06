import json
import argparse
import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, fbeta_score, accuracy_score, confusion_matrix
from sklearn.pipeline import FeatureUnion, Pipeline

def load_data(filepath):
    print(f"Loading {filepath}...")
    texts = []
    labels = []
    ids = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                texts.append(obj['text'])
                labels.append(obj['label'])
                ids.append(obj['id'])
    return pd.DataFrame({'id': ids, 'text': texts, 'label': labels})

def evaluate_model(y_true, y_prob, name="Custom Model"):
    y_pred = (y_prob >= 0.5).astype(int)
    
    roc_auc = roc_auc_score(y_true, y_prob)
    brier = 1.0 - brier_score_loss(y_true, y_prob)
    c_at_1 = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    f05u = fbeta_score(y_true, y_pred, beta=0.5)
    mean_score = np.mean([roc_auc, brier, c_at_1, f1, f05u])
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    print(f"=== Evaluation Results for: {name} ===")
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"Brier   : {brier:.4f}")
    print(f"C@1     : {c_at_1:.4f}")
    print(f"F1      : {f1:.4f}")
    print(f"F0.5u   : {f05u:.4f}")
    print(f"Mean    : {mean_score:.4f}")
    print(f"----------------------")
    print(f"FPR     : {fpr:.4f}")
    print(f"FNR     : {fnr:.4f}")
    print("===========================================\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Train the custom PAN 2026 N-gram detector.")
    parser.add_argument("--train", default="data/train.jsonl", help="Training JSONL path.")
    parser.add_argument("--val", default="data/val.jsonl", help="Validation JSONL path.")
    parser.add_argument(
        "--predictions-out",
        default="custom_model.jsonl",
        help="Where validation predictions are written.",
    )
    parser.add_argument(
        "--model-out",
        default="models/ngram_pipeline.joblib",
        help="Where the serialized sklearn pipeline is written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load data
    train_df = load_data(args.train)
    val_df = load_data(args.val)
    
    print(f"\nTrain size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    
    # We will use a powerful ensemble of Character and Word N-Grams
    print("\nBuilding vectorizer (Word N-grams + Char N-grams)...")
    t0 = time.time()
    
    # The official baseline only uses 1000 word features. We'll use 50,000 word features and 50,000 char features.
    word_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),
        max_features=50000,
        sublinear_tf=True
    )
    
    char_vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        max_features=50000,
        sublinear_tf=True
    )
    
    vectorizer = FeatureUnion([
        ("word", word_vectorizer),
        ("char", char_vectorizer)
    ])
    
    y_train = train_df['label'].values
    y_val = val_df['label'].values

    clf = LogisticRegression(C=10, max_iter=1000, solver='liblinear', random_state=42)
    pipeline = Pipeline([
        ("features", vectorizer),
        ("classifier", clf),
    ])

    # Train Logistic Regression
    # C=10 allows the model to learn more complex patterns without heavily regularizing them
    print("\nTraining Logistic Regression Classifier...")
    pipeline.fit(train_df['text'], y_train)
    print(f"Training and vectorization done in {time.time()-t0:.1f}s")
    
    # Predict and evaluate
    print("\nPredicting on validation set...")
    y_prob = pipeline.predict_proba(val_df['text'])[:, 1]
    
    evaluate_model(y_val, y_prob, name="Advanced N-Gram Logistic Regression")

    print(f"Saving serialized model to {args.model_out}...")
    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    joblib.dump(pipeline, args.model_out)
    
    # Save predictions to file so the user has them!
    print(f"Saving predictions to {args.predictions_out}...")
    with open(args.predictions_out, 'w', encoding='utf-8') as f:
        for i, doc_id in enumerate(val_df['id']):
            f.write(json.dumps({"id": doc_id, "label": float(y_prob[i])}) + "\n")
            
    print("All done!")

if __name__ == '__main__':
    main()
