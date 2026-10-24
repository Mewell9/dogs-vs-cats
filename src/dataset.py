"""Arrange dog and cat photos for flow_from_directory."""

import os
import random
import shutil
import zipfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASSES = ("cat", "dog")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_DIR = PROCESSED_DIR / "train"
TEST_DIR = PROCESSED_DIR / "test"
FIGURES_DIR = PROJECT_ROOT / "figures"
ARCHIVE_NAME = "kagglecatsanddogs.zip"


def label_for(path):
    """Return cat or dog from a parent folder or a name such as dog.12.jpg."""
    names = [path.stem.lower(), *[part.lower() for part in path.parts]]
    for label in CLASSES:
        if any(name == label or name.startswith(f"{label}.") for name in names):
            return label
    return None


def readable_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def extract_archive():
    archive = RAW_DIR / ARCHIVE_NAME
    if not archive.exists():
        raise FileNotFoundError(
            f"Missing {archive}. Download the cats and dogs photos into data/raw."
        )
    marker = RAW_DIR / "PetImages"
    if marker.exists():
        return
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(RAW_DIR)


def discover_images(root=RAW_DIR):
    found = {label: [] for label in CLASSES}
    if not root.exists():
        return found
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if "processed" in path.parts:
            continue
        label = label_for(path)
        if label and readable_image(path):
            found[label].append(path)
    for paths in found.values():
        paths.sort()
    return found


def _place(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def arrange_split(images, train_ratio=0.8, seed=42):
    """Put photos in train and test folders, with one subdirectory per class."""
    random.seed(seed)
    counts = {}
    for label, paths in images.items():
        ordered = list(paths)
        random.shuffle(ordered)
        cut = int(len(ordered) * train_ratio)
        groups = {"train": ordered[:cut], "test": ordered[cut:]}
        for split, split_paths in groups.items():
            folder = (TRAIN_DIR if split == "train" else TEST_DIR) / label
            for index, source in enumerate(split_paths):
                _place(source, folder / f"{label}.{index}{source.suffix.lower()}")
            counts[f"{split}_{label}"] = len(split_paths)
    return counts


def split_ready():
    return all(
        (folder / label).exists() and any((folder / label).iterdir())
        for folder in (TRAIN_DIR, TEST_DIR)
        for label in CLASSES
    )
