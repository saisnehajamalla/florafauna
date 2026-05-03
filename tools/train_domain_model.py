import argparse
import os
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def find_train_test_dirs(domain_key):
    if domain_key == 'plant':
        train_dir = Path('data/train/plant_diseases/PlantVillage')
        test_dir = Path('data/test/plant_diseases/PlantVillage')
        if not train_dir.exists():
            train_dir = Path('data/train/plant_diseases')
        if not test_dir.exists():
            test_dir = Path('data/test/plant_diseases')
    else:
        train_dir = Path(f'data/train/{domain_key}_diseases')
        test_dir = Path(f'data/test/{domain_key}_diseases')
    return train_dir, test_dir


def build_and_train(domain_key, epochs=10, batch_size=32):
    train_dir, test_dir = find_train_test_dirs(domain_key)
    if not train_dir.exists():
        print('Train directory not found:', train_dir)
        return
    if not test_dir.exists():
        print('Test/validation directory not found (continuing without validation):', test_dir)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_data = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical'
    )

    if test_dir.exists():
        test_data = test_datagen.flow_from_directory(
            str(test_dir),
            target_size=(128, 128),
            batch_size=batch_size,
            class_mode='categorical'
        )
    else:
        test_data = None

    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(train_data.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=test_data
    )

    out_dir = Path('models')
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{domain_key}_disease_model.h5"
    model.save(str(model_path))
    print('Saved model to', model_path)

    # save class indices
    class_index_name = f'class_indices_{domain_key}.json'
    if domain_key == 'plant':
        # keep legacy file name for plant to not break things
        class_index_name = 'class_indices.json'
    ci_path = out_dir / class_index_name
    with open(ci_path, 'w', encoding='utf-8') as f:
        json.dump(train_data.class_indices, f, indent=2)
    print('Saved class indices to', ci_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', choices=['plant', 'animal'], default='plant')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    build_and_train(args.domain, epochs=args.epochs, batch_size=args.batch_size)
