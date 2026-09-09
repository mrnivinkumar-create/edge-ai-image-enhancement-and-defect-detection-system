from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# -----------------------------
# Project Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"

MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "model.keras"


IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 30

def preprocess_for_training(image: np.ndarray) -> np.ndarray:
    """
    Image enhancement pipeline.
    """

    image = cv2.resize(
        image.astype(np.uint8),
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    image = cv2.GaussianBlur(image, (3, 3), 0)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    l = clahe.apply(l)

    enhanced = cv2.cvtColor(
        cv2.merge((l, a, b)),
        cv2.COLOR_LAB2RGB,
    )

    enhanced = cv2.normalize(
        enhanced,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    enhanced = enhanced.astype(np.float32) / 255.0

    return enhanced

def build_model():

    model = keras.Sequential(

        [

            layers.Input(shape=(224,224,3)),

            layers.Conv2D(32,3,activation="relu",padding="same"),
            layers.MaxPooling2D(),

            layers.Conv2D(64,3,activation="relu",padding="same"),
            layers.MaxPooling2D(),

            layers.Conv2D(128,3,activation="relu",padding="same"),
            layers.MaxPooling2D(),

            layers.Conv2D(256,3,activation="relu",padding="same"),
            layers.MaxPooling2D(),

            layers.GlobalAveragePooling2D(),

            layers.Dense(256,activation="relu"),

            layers.Dropout(0.5),

            layers.Dense(6,activation="softmax")

        ]

    )

    model.compile(

        optimizer=keras.optimizers.Adam(
            learning_rate=0.0001
        ),

        loss="categorical_crossentropy",

        metrics=["accuracy"]

    )

    return model

def get_generators():

    train_datagen = ImageDataGenerator(

        preprocessing_function=preprocess_for_training,

        rotation_range=10,

        width_shift_range=0.1,

        height_shift_range=0.1,

        zoom_range=0.1,

        horizontal_flip=True,

    )

    validation_datagen = ImageDataGenerator(

        preprocessing_function=preprocess_for_training

    )

    train_generator = train_datagen.flow_from_directory(

        DATASET_DIR / "train",

        target_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        class_mode="categorical",

        shuffle=True,

    )

    validation_generator = validation_datagen.flow_from_directory(

        DATASET_DIR / "validation",

        target_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        class_mode="categorical",

        shuffle=False,

    )

    return train_generator, validation_generator

def save_graphs(history):

    MODEL_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(8,5))

    plt.plot(history.history["accuracy"],label="Training")

    plt.plot(history.history["val_accuracy"],label="Validation")

    plt.title("Accuracy")

    plt.legend()

    plt.savefig(MODEL_DIR/"accuracy.png")

    plt.close()

    plt.figure(figsize=(8,5))

    plt.plot(history.history["loss"],label="Training")

    plt.plot(history.history["val_loss"],label="Validation")

    plt.title("Loss")

    plt.legend()

    plt.savefig(MODEL_DIR/"loss.png")

    plt.close()

def main():

    MODEL_DIR.mkdir(exist_ok=True)

    train_generator, validation_generator = get_generators()

    model = build_model()

    callbacks=[

        keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=5,

            restore_best_weights=True

        ),

        keras.callbacks.ModelCheckpoint(

            MODEL_PATH,

            save_best_only=True,

            monitor="val_accuracy"

        )

    ]

    history=model.fit(

        train_generator,

        validation_data=validation_generator,

        epochs=EPOCHS,

        callbacks=callbacks

    )

    save_graphs(history)

    class_names={

        index:name

        for name,index in train_generator.class_indices.items()

    }

    with open(MODEL_DIR/"class_names.json","w") as f:

        json.dump(class_names,f,indent=4)

    print("\nTraining Finished")

    print("Model Saved:",MODEL_PATH)

    test_dir = DATASET_DIR / "test"

    if test_dir.exists():

        test_datagen = ImageDataGenerator(

            preprocessing_function=preprocess_for_training

        )

        test_generator = test_datagen.flow_from_directory(

            test_dir,

            target_size=IMAGE_SIZE,

            batch_size=BATCH_SIZE,

            class_mode="categorical",

            shuffle=False,

        )

        loss, accuracy = model.evaluate(test_generator)

        print("\nTest Accuracy:", accuracy)

        predictions = model.predict(test_generator)

        predicted = np.argmax(predictions, axis=1)

        report = classification_report(

            test_generator.classes,

            predicted,

            target_names=list(class_names.values())

        )

        with open(MODEL_DIR/"classification_report.txt","w") as f:

            f.write(report)

        matrix = confusion_matrix(

            test_generator.classes,

            predicted

        )

        disp = ConfusionMatrixDisplay(

            confusion_matrix=matrix,

            display_labels=list(class_names.values())

        )

        disp.plot()

        plt.savefig(MODEL_DIR/"confusion_matrix.png")

        plt.close()

if __name__ == "__main__":
    main()