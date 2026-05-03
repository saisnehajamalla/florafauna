import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import re
import joblib
import os
import json
import uuid
from pathlib import Path
from tools import db as db_module
from scipy.ndimage import rotate

# -------- Configuration --------
# We'll support multiple domains (plant vs animal). The app will look for domain-specific
# artifacts and fall back to reasonable defaults for the plant model which already exists.
# Domain keys: 'plant' and 'animal'

# Helper to build domain-specific paths
def paths_for_domain(domain_key):
    # domain_key in {'plant', 'animal'}
    if domain_key == 'plant':
        return {
            'model': 'models/plant_disease_model.h5',
            'class_index': 'models/class_indices.json',
            'temperature': 'models/temperature.txt',
            'disease_info': 'data/disease_info.json'
        }
    else:
        # animal domain: user is expected to add these files when training an animal model
        return {
            'model': 'models/animal_disease_model.h5',
            'class_index': 'models/class_indices_animal.json',
            'temperature': 'models/temperature_animal.txt',
            'disease_info': 'data/disease_info_animal.json'
        }

# Model cache to avoid re-loading on every rerun
MODEL_CACHE = {}

# Load class names reliably: prefer saved mapping, else fallback to directory listing
def load_class_names(domain_key='plant'):
    paths = paths_for_domain(domain_key)
    class_index_path = paths.get('class_index')
    # prefer saved mapping
    if class_index_path and os.path.exists(class_index_path):
        try:
            with open(class_index_path, 'r', encoding='utf-8') as f:
                class_indices = json.load(f)  # {class_name: index}
            inv = {int(v): k for k, v in class_indices.items()}
            return [inv[i] for i in range(len(inv))]
        except Exception:
            pass

    # fallback: look for domain-specific train folders
    if domain_key == 'plant':
        fallback_dir = "data/train/plant_diseases/PlantVillage"
        if os.path.exists(fallback_dir):
            return sorted([d for d in os.listdir(fallback_dir) if os.path.isdir(os.path.join(fallback_dir, d))])
        fallback_dir2 = "data/train/plant_diseases"
        if os.path.exists(fallback_dir2):
            subdirs = []
            for root, dirs, files in os.walk(fallback_dir2):
                subdirs.extend(dirs)
                break
            return sorted(subdirs)
    else:
        fallback_dir = "data/train/animal_diseases"
        if os.path.exists(fallback_dir):
            return sorted([d for d in os.listdir(fallback_dir) if os.path.isdir(os.path.join(fallback_dir, d))])

    # last resort: try data/train
    fallback_dir3 = "data/train"
    if os.path.exists(fallback_dir3):
        return sorted([d for d in os.listdir(fallback_dir3) if os.path.isdir(os.path.join(fallback_dir3, d))])

    return []


def get_model_for_domain(domain_key='plant'):
    paths = paths_for_domain(domain_key)
    model_path = paths.get('model')
    if model_path in MODEL_CACHE:
        return MODEL_CACHE[model_path]
    if model_path and os.path.exists(model_path):
        try:
            m = tf.keras.models.load_model(model_path)
            MODEL_CACHE[model_path] = m
            return m
        except Exception:
            return None
    return None



# Note: CLASS_NAMES will be loaded after domain selection


def load_temperature():
    # return float temperature if file exists, else None
    paths = paths_for_domain(domain_key)
    temp_path = paths.get('temperature')
    if temp_path and os.path.exists(temp_path):
        try:
            with open(temp_path, 'r', encoding='utf-8') as f:
                t = float(f.read().strip())
                if t > 0.0:
                    return t
        except Exception:
            pass
    return None


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def apply_temperature_scaling(probs, temperature):
    # probs: 1D numpy array of probabilities (sum ~1)
    # approximate logits by log probs (clip to avoid -inf), then rescale
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    logits = np.log(probs)
    scaled = softmax(logits / temperature)
    return scaled


st.set_page_config(page_title="FloraFauna AI", layout="wide")
st.title("🌿 FloraFauna AI - Flora & Fauna Disease Detector")
st.write("Upload an image to detect possible diseases. Choose Plant or Animal domain in the sidebar.")

# Domain selector in the sidebar
domain_label = st.sidebar.selectbox("Domain", ["Plant diseases", "Animal diseases"]) 
domain_key = 'plant' if domain_label.startswith('Plant') else 'animal'

# Load domain-specific resources
paths = paths_for_domain(domain_key)
# disease info (domain-specific)
DISEASE_INFO = {}
di_path = paths.get('disease_info')
if di_path and Path(di_path).exists():
    try:
        with open(di_path, 'r', encoding='utf-8') as f:
            DISEASE_INFO = json.load(f)
    except Exception:
        DISEASE_INFO = {}

# class names and model (lazy-loaded)
CLASS_NAMES = load_class_names(domain_key)
model = get_model_for_domain(domain_key)

# Load symptom-based model (if present) for animal domain
symptom_model = None
symptom_vectorizer = None
symptom_model_path = 'models/animal_symptom_model.pkl'
symptom_vec_path = 'models/animal_symptom_vectorizer.pkl'
if domain_key == 'animal':
    if os.path.exists(symptom_model_path):
        try:
            symptom_model = joblib.load(symptom_model_path)
        except Exception:
            symptom_model = None
    if os.path.exists(symptom_vec_path):
        try:
            symptom_vectorizer = joblib.load(symptom_vec_path)
        except Exception:
            symptom_vectorizer = None

def _clean_symptom_text(s: str):
    if not s:
        return ''
    s = s.lower()
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# Provide a small notice when model or classes are missing
if model is None:
    st.sidebar.warning(f"No model found for domain '{domain_label}'. Place a model at: {paths.get('model')}")
if not CLASS_NAMES:
    st.sidebar.info(f"No class folders or class index found for domain '{domain_label}'. Add training data under data/train/{domain_key}_diseases or generate class_indices file.")

# Upload an image (only shown if an image model is available for the selected domain)
uploaded_file = None
if model is not None:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"]) 

if uploaded_file is not None:
    # Display image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # save uploaded file to disk (for logging and later review)
    uploads_dir = Path("models/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
    image_path = uploads_dir / filename
    with open(image_path, 'wb') as out:
        out.write(uploaded_file.getbuffer())

    # Preprocess image (must match training preprocessing)
    target_size = (128, 128)
    img = image.load_img(uploaded_file, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    # Check model availability (should not happen because uploader shown only if model exists)
    if model is None:
        st.error(f"No model available for domain '{domain_label}'. Please add a model file at: {paths.get('model')}")
        st.stop()

    # Predict
    raw_preds = model.predict(img_array)
    raw = raw_preds[0]

    # If model outputs logits (not probabilities), convert to probs, else keep
    # Determine whether outputs look like probabilities (sum close to 1)
    if np.all(raw >= 0) and (abs(np.sum(raw) - 1.0) < 1e-3):
        probs = raw
    else:
        # assume logits
        probs = softmax(raw)

    # Apply temperature scaling if available
    temperature = load_temperature(domain_key)
    if temperature is not None:
        calibrated = apply_temperature_scaling(probs, temperature)
    else:
        calibrated = probs

    # Data augmentation: flip and rotate (TTA)
    imgs = [img_array[0],
            np.flip(img_array[0], axis=2),
            rotate(img_array[0], 10, reshape=False),
            rotate(img_array[0], -10, reshape=False)]
    # Get predictions for each augmented image and stack into shape (n_aug, num_classes)
    preds_list = [model.predict(np.expand_dims(im, axis=0))[0] for im in imgs]
    preds_arr = np.stack(preds_list, axis=0)
    avg_logits_or_probs = np.mean(preds_arr, axis=0)  # 1D array of length num_classes

    # Ensure we have probabilities (not logits)
    if not (np.all(avg_logits_or_probs >= 0) and abs(np.sum(avg_logits_or_probs) - 1.0) < 1e-3):
        avg_probs = softmax(avg_logits_or_probs)
    else:
        avg_probs = avg_logits_or_probs

    # Apply temperature scaling if available
    temperature = load_temperature(domain_key)
    if temperature is not None:
        avg_probs = apply_temperature_scaling(avg_probs, temperature)

    # Show top-k
    top_k = min(3, len(avg_probs)) if len(avg_probs) > 0 else 0
    top_idx = np.argsort(avg_probs)[-top_k:][::-1]

    if len(CLASS_NAMES) != 0 and len(CLASS_NAMES) == len(avg_probs):
        st.subheader("Predictions")
        top_k_dict = {}
        cols = st.columns([2,1])
        with cols[0]:
            for i in top_idx:
                label = CLASS_NAMES[i]
                conf = 100.0 * avg_probs[i]
                top_k_dict[label] = float(conf)
                st.write(f"{label}: {conf:.2f}%")

        # show tips / recommendations in right column
        with cols[1]:
            primary = CLASS_NAMES[top_idx[0]]
            info = DISEASE_INFO.get(primary)
            st.markdown("**Recommendations**")
            if info:
                st.markdown(f"**Pesticide / Remedy:** {info.get('pesticide_recommendation', 'N/A')}")
                st.markdown(f"**Care tips:** {info.get('care_tips', 'N/A')}")
            else:
                st.markdown("No specific recommendations available for this class.")

        # log prediction to sqlite DB (fire-and-forget)
        try:
            db_module.log_prediction(str(image_path), primary, float(top_k_dict.get(primary, 0.0))/100.0, top_k_dict, temperature)
        except Exception as e:
            st.write("(Warning) Could not log prediction:", e)
    else:
        # fallback: show indices
        st.subheader("Predictions (indices)")
        for i in top_idx:
            conf = 100.0 * avg_probs[i]
            st.write(f"Class {i}: {conf:.2f}%")

    if temperature is not None:
        st.info(f"Applied temperature scaling with T={temperature:.3f}")

# ---------------------
# Symptom-based prediction UI for Animal domain
# ---------------------
if domain_key == 'animal':
    st.subheader('Animal symptom-based diagnosis')
    st.markdown('If you don\'t have an animal image model, enter symptoms (comma separated) below and press **Predict from symptoms**.')
    symptoms_text = st.text_area('Symptoms (comma-separated)', value='')
    if st.button('Predict from symptoms'):
        if symptom_model is None or symptom_vectorizer is None:
            st.warning('No symptom model/vectorizer found. Train one with: python tools/train_animal_symptom_model.py')
        else:
            clean = _clean_symptom_text(symptoms_text)
            X = symptom_vectorizer.transform([clean])
            probs = symptom_model.predict_proba(X)[0]
            classes = list(symptom_model.classes_)
            top_k = min(3, len(probs))
            top_idx = np.argsort(probs)[-top_k:][::-1]
            st.subheader('Predictions (from symptoms)')
            top_k_dict = {}
            cols = st.columns([2,1])
            with cols[0]:
                for i in top_idx:
                    label = classes[i]
                    conf = 100.0 * probs[i]
                    top_k_dict[label] = float(conf)
                    st.write(f"{label}: {conf:.2f}%")
            with cols[1]:
                primary = classes[top_idx[0]]
                info = DISEASE_INFO.get(primary)
                st.markdown('**Recommendations**')
                if info:
                    st.markdown(f"**Pesticide / Remedy:** {info.get('pesticide_recommendation', 'N/A')}")
                    st.markdown(f"**Care tips:** {info.get('care_tips', 'N/A')}")
                else:
                    st.markdown('No specific recommendations available for this class.')

            # log prediction
            try:
                db_module.log_prediction('symptom_input', primary, float(top_k_dict.get(primary, 0.0))/100.0, top_k_dict, None)
            except Exception as e:
                st.write('(Warning) Could not log symptom prediction:', e)

# Display recent predictions from DB
st.sidebar.header("Recent Predictions")
try:
    rows = db_module.fetch_recent_predictions(10)  # fetch top 10 recent predictions
    if rows:
        for row in rows:
            # unpack row data (id, timestamp, predicted_class, confidence, top_k_json)
            id, timestamp, predicted_class, confidence, top_k_json = row
            st.sidebar.write(f"ID: {id}, Class: {predicted_class}, Confidence: {confidence:.2f}%")
except Exception as e:
    st.sidebar.write("Error loading predictions from database:", e)

# Additional hardcoded disease info (for testing / fallback)
# (removed long embedded JSON to avoid syntax errors; use data/disease_info.json instead)