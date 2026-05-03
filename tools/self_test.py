import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from scipy.ndimage import rotate
from pathlib import Path
from tools import db as db_module

MODEL_PATH = "models/plant_disease_model.h5"
CLASS_INDEX_PATH = "models/class_indices.json"
TEMPERATURE_PATH = "models/temperature.txt"
TEST_DIR = "data/test/plant_diseases/PlantVillage"
TARGET_SIZE = (128, 128)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def load_class_names():
    if os.path.exists(CLASS_INDEX_PATH):
        try:
            with open(CLASS_INDEX_PATH, 'r', encoding='utf-8') as f:
                class_indices = json.load(f)
            inv = {int(v): k for k, v in class_indices.items()}
            return [inv[i] for i in range(len(inv))]
        except Exception:
            pass
    if os.path.exists(TEST_DIR):
        return sorted([d for d in os.listdir(os.path.dirname(TEST_DIR)) if os.path.isdir(os.path.join(os.path.dirname(TEST_DIR), d))])
    if os.path.exists('data/train'):
        return sorted([d for d in os.listdir('data/train') if os.path.isdir(os.path.join('data/train', d))])
    return []


def load_temperature():
    if os.path.exists(TEMPERATURE_PATH):
        try:
            return float(open(TEMPERATURE_PATH).read().strip())
        except Exception:
            return None
    return None


def find_sample_image():
    # Search TEST_DIR for first image file
    for root, dirs, files in os.walk(TEST_DIR):
        for fn in files:
            if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                return os.path.join(root, fn)
    # fallback: search data/test recursively
    for root, dirs, files in os.walk('data/test'):
        for fn in files:
            if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                return os.path.join(root, fn)
    return None


def preprocess_image(path):
    img = image.load_img(path, target_size=TARGET_SIZE)
    arr = image.img_to_array(img)
    arr = arr / 255.0
    return arr


if __name__ == '__main__':
    print('Self-test: start')
    if not os.path.exists(MODEL_PATH):
        print('Model not found at', MODEL_PATH)
        raise SystemExit(1)

    model = tf.keras.models.load_model(MODEL_PATH)
    print('Model loaded.')

    class_names = load_class_names()
    print('Number of classes resolved:', len(class_names))

    temperature = load_temperature()
    print('Loaded temperature:', temperature)

    sample = find_sample_image()
    if not sample:
        print('No sample image found under data/test. Please provide an image.')
        raise SystemExit(1)

    print('Using sample image:', sample)
    arr = preprocess_image(sample)

    # TTA
    tta = [arr, np.flip(arr, axis=1), rotate(arr, 10, reshape=False), rotate(arr, -10, reshape=False)]
    preds = []
    for im in tta:
        p = model.predict(np.expand_dims(im, axis=0))[0]
        if not (np.all(p >= 0) and abs(np.sum(p) - 1.0) < 1e-3):
            p = softmax(p)
        preds.append(p)

    avg = np.mean(np.stack(preds, axis=0), axis=0)

    if temperature is not None:
        # approximate logits by log probs and rescale
        eps = 1e-12
        probs = np.clip(avg, eps, 1.0)
        logits = np.log(probs)
        avg = softmax(logits / temperature)

    top_k = 3
    idx = np.argsort(avg)[-top_k:][::-1]
    print('\nTop predictions:')
    top_k_dict = {}
    for i in idx:
        lbl = class_names[i] if i < len(class_names) else f'class_{i}'
        conf = float(avg[i])
        top_k_dict[lbl] = conf
        print(f'  {lbl}: {conf:.4f}')

    # show disease info if present
    DISEASE_INFO_PATH = 'data/disease_info.json'
    if os.path.exists(DISEASE_INFO_PATH):
        try:
            with open(DISEASE_INFO_PATH, 'r', encoding='utf-8') as f:
                disease_info = json.load(f)
        except Exception:
            disease_info = {}
    else:
        disease_info = {}

    primary = list(top_k_dict.keys())[0]
    info = disease_info.get(primary)
    print('\nRecommendations for', primary)
    if info:
        print('Pesticide / Remedy:', info.get('pesticide_recommendation'))
        print('Care tips:', info.get('care_tips'))
    else:
        print('No disease info available for this class.')

    # log into DB
    try:
        db_module.log_prediction(sample, primary, top_k_dict[primary], top_k_dict, temperature)
        print('\nLogged prediction to DB')
    except Exception as e:
        print('Failed to log prediction:', e)

    # fetch recent
    try:
        recent = db_module.fetch_recent_predictions(5)
        print('\nRecent DB entries:')
        for r in recent:
            print(r)
    except Exception as e:
        print('Failed to fetch recent predictions:', e)

    print('\nSelf-test: done')
