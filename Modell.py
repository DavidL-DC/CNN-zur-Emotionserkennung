import os
import numpy as np
import tensorflow as tf
from keras.utils import image_dataset_from_directory
from keras import models, layers
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Pfade zu den Bildordnern
train_dir = "train"
test_dir = "test"

# Trainings- und Validierungsdaten laden
train_data = image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    image_size=(48, 48),
    batch_size=64,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_data = image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    image_size=(48, 48),
    batch_size=64,
    validation_split=0.2,
    subset="validation",
    seed=42
)

# Testdaten (keine Augmentation)
test_data = image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    image_size=(48, 48),
    batch_size=64,
    shuffle=False
)

# Datenaugmentation definieren
data_augmentation = keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

model = models.Sequential([
    layers.Input(shape=(48, 48, 1)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(7, activation='softmax')  # 7 Emotionen
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data.map(lambda x, y: (data_augmentation(x), y)),  # Augmentation anwenden
    validation_data=val_data,
    epochs=25
)

# Testdaten normalisieren (falls nötig)
test_data = test_data.map(lambda x, y: (x / 255.0, y))

# Vorhersagen erzeugen
predictions = model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)

# Wahre Klassen
true_classes = np.concatenate([y.numpy() for x, y in test_data])
true_classes = np.argmax(true_classes, axis=1)

# Confusion-Matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Klassifikationsbericht
report = classification_report(true_classes, predicted_classes, target_names=train_data.class_names)
print(report)

model.save("model.keras")