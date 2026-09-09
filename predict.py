"""Image preprocessing and six-class inference utilities."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
IMAGE_SIZE = (224, 224)
EXPECTED_CLASS_COUNT = 6


class ModelUnavailableError(RuntimeError):
    """Raised when trained model assets are missing or incompatible."""


def enhance_image(image_path: str | Path) -> np.ndarray:
    """Read, enhance, resize, and normalize an input image for inference."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a readable image.")

    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    enhanced = cv2.cvtColor(
        cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR
    )
    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


@lru_cache(maxsize=1)
def load_class_names() -> list[str]:
    """Load class names from class_names.json."""

    if not CLASS_NAMES_PATH.exists():
        raise ModelUnavailableError(
            "class_names.json is missing. Run train.py again."
        )

    try:
        names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    except Exception as error:
        raise ModelUnavailableError(
            "Unable to read class_names.json."
        ) from error

    # Supports dictionary format:
    # {"0":"crazing","1":"inclusion",...}
    if isinstance(names, dict):
        names = [names[str(i)] for i in range(EXPECTED_CLASS_COUNT)]

    if len(names) != EXPECTED_CLASS_COUNT:
        raise ModelUnavailableError(
            "Invalid class_names.json."
        )

    return names


@lru_cache(maxsize=1)
def load_model() -> keras.Model:
    """Load and cache the native Keras model format supported by Keras 3."""
    if not MODEL_PATH.exists():
        raise ModelUnavailableError("No trained model was found. Run train.py first.")
    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
    except (OSError, ValueError) as error:
        raise ModelUnavailableError("The trained model could not be loaded. Run train.py again.") from error
    if model.output_shape[-1] != EXPECTED_CLASS_COUNT:
        raise ModelUnavailableError("The model output does not match the six configured classes.")
    return model


def predict_image(image_path: str | Path) -> tuple[str, float, dict[str, float]]:
    """Return the highest-scoring defect class, its confidence, and all probabilities."""
    image_path = Path(image_path)
    image = enhance_image(image_path)
    
    # Save the enhanced image for UI side-by-side comparison
    enhanced_path = image_path.parent / f"enhanced_{image_path.name}"
    # Convert normalized RGB [0,1] back to BGR [0,255] for saving
    save_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(enhanced_path), save_image)
    
    probabilities = np.asarray(load_model().predict(np.expand_dims(image, axis=0), verbose=0)[0])
    class_index = int(np.argmax(probabilities))
    class_names = load_class_names()
    
    all_probs = {class_names[i]: round(float(probabilities[i]) * 100, 2) for i in range(len(class_names))}
    
    return class_names[class_index], round(float(probabilities[class_index]) * 100, 2), all_probs
