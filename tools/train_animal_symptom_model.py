import json
import re
from pathlib import Path
import argparse
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score


def load_csv_rows(paths):
    # prefer train CSV if present, else test CSV
    rows = []
    for p in paths:
        p = Path(p)
        if p.exists():
            with p.open('r', encoding='utf-8') as f:
                # naive CSV parsing: split lines and comma - rely on existing clean CSV
                import csv
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
    return rows


def build_text_and_labels(rows):
    texts = []
    labels = []
    for r in rows:
        label = r.get('AnimalName') or r.get('Animal') or r.get('animal')
        if not label:
            continue
        # join symptom columns
        parts = []
        for k in ['symptoms1','symptoms2','symptoms3','symptoms4','symptoms5']:
            v = r.get(k) or ''
            parts.append(v)
        text = ' '.join(parts)
        # normalize
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        texts.append(text)
        labels.append(label.strip())
    return texts, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csvs', nargs='*', default=['data/test/animal_diseases/data.csv'])
    parser.add_argument('--out-model', default='models/animal_symptom_model.pkl')
    parser.add_argument('--out-vectorizer', default='models/animal_symptom_vectorizer.pkl')
    parser.add_argument('--out-class-index', default='models/class_indices_animal.json')
    args = parser.parse_args()

    rows = load_csv_rows(args.csvs)
    if not rows:
        print('No CSV rows found in', args.csvs)
        return

    texts, labels = build_text_and_labels(rows)
    if not texts:
        print('No texts extracted from CSV')
        return

    # vectorize
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=2000)
    X = vec.fit_transform(texts)
    y = np.array(labels)

    # split (try stratified split; if classes have single samples, fall back to non-stratified)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    acc = accuracy_score(y_test, preds)
    print('Validation accuracy:', acc)
    print(classification_report(y_test, preds))

    # save model and vectorizer
    out_dir = Path('models')
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.out_model)
    joblib.dump(vec, args.out_vectorizer)
    print('Saved model to', args.out_model)

    # save class indices mapping (label -> index)
    classes = {c: i for i, c in enumerate(clf.classes_)}
    with open(args.out_class_index, 'w', encoding='utf-8') as f:
        json.dump(classes, f, indent=2)
    print('Saved class indices to', args.out_class_index)


if __name__ == '__main__':
    main()
