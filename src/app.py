"""Train the dog and cat image classifier and save the model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dataset import (
    CLASSES,
    FIGURES_DIR,
    TEST_DIR,
    TRAIN_DIR,
    arrange_split,
    discover_images,
    extract_archive,
    split_ready,
)
from model import evaluate_and_save, build_model, make_generators, show_test_predictions, train


def save_class_preview(paths, title, destination):
    """Save the first 100 photos of one class in a single figure."""
    selected = paths[:100]
    figure, axes = plt.subplots(10, 10, figsize=(12, 12))
    for axis, path in zip(axes.ravel(), selected):
        with Image.open(path) as image:
            axis.imshow(np.asarray(image.convert("RGB")))
        axis.axis("off")
    for axis in axes.ravel()[len(selected) :]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination)
    plt.close(figure)
    print(f"Saved {destination}")


def main():
    extract_archive()
    images = discover_images()
    for label in CLASSES:
        print(f"{label}: {len(images[label])} readable photos")
        if not images[label]:
            raise SystemExit(f"No {label} photos were found in data/raw.")

    save_class_preview(images["dog"], "First 100 dog photos", FIGURES_DIR / "dogs_first_100.png")
    save_class_preview(images["cat"], "First 100 cat photos", FIGURES_DIR / "cats_first_100.png")

    if not split_ready():
        counts = arrange_split(images)
        print("Split counts:", counts)
    else:
        print("Train and test folders already exist.")

    train_data, test_data = make_generators(TRAIN_DIR, TEST_DIR)
    print("Class indices:", train_data.class_indices)
    model = build_model()
    model.summary()
    train(model, train_data, test_data, epochs=1)
    loss, accuracy = evaluate_and_save(test_data)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("Saved model to models/dogs_vs_cats.keras")
    show_test_predictions(TEST_DIR)


if __name__ == "__main__":
    main()
