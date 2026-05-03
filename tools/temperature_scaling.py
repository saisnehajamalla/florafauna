import json
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

import argparse

# Config defaults (can be overridden with --domain)
TARGET_SIZE = (128, 128)
BATCH_SIZE = 32


def paths_for_domain(domain_key):
    if domain_key == 'plant':
        return {
            'model': 'models/plant_disease_model.h5',
            'test_dir': 'data/test/plant_diseases/PlantVillage',
            'class_index_out': 'models/class_indices.json',
            'temperature_out': 'models/temperature.txt'
        }
    else:
        return {
            'model': 'models/animal_disease_model.h5',
            'test_dir': 'data/test/animal_diseases',
            'class_index_out': 'models/class_indices_animal.json',
            'temperature_out': 'models/temperature_animal.txt'
        }


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def apply_temperature(probs, T):
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    logits = np.log(probs)
    scaled = softmax(logits / T)
    return scaled


def nll(probs, y_true_onehot, T):
    calibrated = apply_temperature(probs, T)
    eps = 1e-12
    return -np.mean(np.sum(y_true_onehot * np.log(np.clip(calibrated, eps, 1.0)), axis=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', choices=['plant', 'animal'], default='plant')
    args = parser.parse_args()
    paths = paths_for_domain(args.domain)
    MODEL_PATH = paths['model']
    TEST_DIR = paths['test_dir']
    CLASS_INDEX_OUT = paths['class_index_out']
    TEMPERATURE_OUT = paths['temperature_out']

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Model not found at {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)

    test_datagen = ImageDataGenerator(rescale=1.0/255)
    test_data = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Save class indices mapping for the app to load later
    class_indices = test_data.class_indices
    os.makedirs(os.path.dirname(CLASS_INDEX_OUT), exist_ok=True)
    with open(CLASS_INDEX_OUT, 'w') as f:
        json.dump(class_indices, f)
    print(f"Saved class indices mapping to {CLASS_INDEX_OUT}")

    # Get predictions on validation/test set
    preds = model.predict(test_data, verbose=1)
    # If outputs look like logits (not sum to 1), convert
    if not (np.all(preds >= 0) and np.allclose(np.sum(preds, axis=1), 1.0, atol=1e-3)):
        probs = softmax(preds)
    else:
        probs = preds

    # true labels one-hot
    y_true = test_data.classes
    num_classes = preds.shape[1]
    y_onehot = np.zeros((len(y_true), num_classes), dtype=np.int32)
    y_onehot[np.arange(len(y_true)), y_true] = 1

    # Grid search for temperature
    ts = np.linspace(0.1, 5.0, 100)
    nlls = [nll(probs, y_onehot, t) for t in ts]
    best_idx = int(np.argmin(nlls))
    best_t = ts[best_idx]

    # refine around best
    ts2 = np.linspace(max(0.01, best_t - 0.5), best_t + 0.5, 100)
    nlls2 = [nll(probs, y_onehot, t) for t in ts2]
    best_t2 = ts2[int(np.argmin(nlls2))]

    os.makedirs(os.path.dirname(TEMPERATURE_OUT), exist_ok=True)
    with open(TEMPERATURE_OUT, 'w') as f:
        f.write(str(float(best_t2)))
    print(f"Saved temperature {best_t2:.6f} to {TEMPERATURE_OUT}")

    # Print basic metrics before/after
    from sklearn.metrics import log_loss, accuracy_score

    pred_labels = np.argmax(probs, axis=1)
    acc_before = accuracy_score(y_true, pred_labels)
    ll_before = log_loss(y_onehot, probs)

    calibrated = apply_temperature(probs, best_t2)
    pred_labels_after = np.argmax(calibrated, axis=1)
    acc_after = accuracy_score(y_true, pred_labels_after)
    ll_after = log_loss(y_onehot, calibrated)

    print(f"Acc before: {acc_before:.4f}, NLL before: {ll_before:.4f}")
    print(f"Acc after:  {acc_after:.4f}, NLL after:  {ll_after:.4f}")
    print("Done.")
