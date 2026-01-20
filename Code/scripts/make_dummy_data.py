#!/usr/bin/env python3
"""
Generate dummy data for testing the Adaptive-DINO-ICD pipeline.

Creates:
- DISC-style folder with synthetic images
- NDEC-style pairs CSV with labels

Run with:
    python scripts/make_dummy_data.py [--output_dir ./data]
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def generate_random_image(
    size: Tuple[int, int] = (224, 224),
    pattern: str = "random",
) -> Image.Image:
    """
    Generate a random synthetic image.

    Args:
        size: Image size (width, height)
        pattern: Type of pattern ("random", "gradient", "shapes", "texture")

    Returns:
        PIL Image
    """
    if pattern == "random":
        # Random noise
        arr = np.random.randint(0, 256, (*size[::-1], 3), dtype=np.uint8)
        return Image.fromarray(arr)

    elif pattern == "gradient":
        # Gradient image
        arr = np.zeros((*size[::-1], 3), dtype=np.uint8)
        for i in range(size[1]):
            for j in range(size[0]):
                arr[i, j] = [
                    int(255 * i / size[1]),
                    int(255 * j / size[0]),
                    int(255 * (i + j) / (size[0] + size[1])),
                ]
        return Image.fromarray(arr)

    elif pattern == "shapes":
        # Random shapes
        img = Image.new("RGB", size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        for _ in range(random.randint(3, 8)):
            shape = random.choice(["rectangle", "ellipse", "line"])
            color = tuple(random.randint(0, 255) for _ in range(3))
            x1, y1 = random.randint(0, size[0]), random.randint(0, size[1])
            x2, y2 = random.randint(0, size[0]), random.randint(0, size[1])

            if shape == "rectangle":
                draw.rectangle([x1, y1, x2, y2], fill=color)
            elif shape == "ellipse":
                draw.ellipse([x1, y1, x2, y2], fill=color)
            else:
                draw.line([x1, y1, x2, y2], fill=color, width=3)

        return img

    elif pattern == "texture":
        # Textured image with blur
        arr = np.random.randint(0, 256, (*size[::-1], 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
        return img

    elif pattern == "uniform":
        # Uniform color (low entropy)
        color = tuple(random.randint(100, 200) for _ in range(3))
        return Image.new("RGB", size, color=color)

    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def generate_disc_data(
    output_dir: Path,
    num_images: int = 100,
    image_size: Tuple[int, int] = (256, 256),
) -> None:
    """
    Generate DISC-style folder with images.

    Args:
        output_dir: Output directory for DISC images
        num_images: Number of images to generate
        image_size: Size of generated images
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    patterns = ["random", "gradient", "shapes", "texture", "uniform"]

    logger.info(f"Generating {num_images} DISC images in {output_dir}")

    for i in range(num_images):
        pattern = random.choice(patterns)
        img = generate_random_image(image_size, pattern)

        # Save image
        img_path = output_dir / f"disc_{i:05d}.jpg"
        img.save(img_path, "JPEG", quality=95)

        if (i + 1) % 20 == 0:
            logger.info(f"Generated {i + 1}/{num_images} images")

    logger.info(f"DISC data generation complete: {num_images} images")


def generate_ndec_data(
    output_dir: Path,
    num_pairs: int = 50,
    pos_ratio: float = 0.6,
    image_size: Tuple[int, int] = (256, 256),
) -> None:
    """
    Generate NDEC-style pairs with annotations.

    Args:
        output_dir: Output directory for NDEC data
        num_pairs: Number of image pairs to generate
        pos_ratio: Ratio of positive (copy) pairs
        image_size: Size of generated images
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    patterns = ["random", "gradient", "shapes", "texture"]

    logger.info(f"Generating {num_pairs} NDEC pairs in {output_dir}")

    for i in range(num_pairs):
        is_positive = random.random() < pos_ratio

        if is_positive:
            # Generate original and modified copy
            pattern = random.choice(patterns)
            img_a = generate_random_image(image_size, pattern)

            # Create copy with modifications
            img_b = img_a.copy()

            # Apply random modifications
            modifications = random.sample(
                ["crop", "rotate", "blur", "color", "resize"],
                k=random.randint(1, 3)
            )

            for mod in modifications:
                if mod == "crop":
                    # Random crop
                    w, h = img_b.size
                    crop_ratio = random.uniform(0.6, 0.9)
                    new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
                    x = random.randint(0, w - new_w)
                    y = random.randint(0, h - new_h)
                    img_b = img_b.crop((x, y, x + new_w, y + new_h))
                    img_b = img_b.resize(image_size)

                elif mod == "rotate":
                    angle = random.uniform(-30, 30)
                    img_b = img_b.rotate(angle, expand=False, fillcolor=(128, 128, 128))

                elif mod == "blur":
                    img_b = img_b.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2)))

                elif mod == "color":
                    # Adjust brightness/contrast
                    from PIL import ImageEnhance
                    enhancer = ImageEnhance.Brightness(img_b)
                    img_b = enhancer.enhance(random.uniform(0.8, 1.2))

                elif mod == "resize":
                    scale = random.uniform(0.7, 1.3)
                    new_size = (int(image_size[0] * scale), int(image_size[1] * scale))
                    img_b = img_b.resize(new_size)
                    img_b = img_b.resize(image_size)

            label = "pos"
            direction = "a->b"  # a is original, b is copy
            similar_for_metric = True

        else:
            # Generate two different images
            pattern_a = random.choice(patterns)
            pattern_b = random.choice(patterns)
            img_a = generate_random_image(image_size, pattern_a)
            img_b = generate_random_image(image_size, pattern_b)

            label = "neg"
            direction = "a->b"
            similar_for_metric = False

        # Save images
        path_a = f"pair_{i:04d}_a.jpg"
        path_b = f"pair_{i:04d}_b.jpg"

        img_a.save(images_dir / path_a, "JPEG", quality=95)
        img_b.save(images_dir / path_b, "JPEG", quality=95)

        pairs.append({
            "img_a_path": path_a,
            "img_b_path": path_b,
            "label": label,
            "direction": direction,
            "similar_pair_for_metric": similar_for_metric,
        })

        if (i + 1) % 10 == 0:
            logger.info(f"Generated {i + 1}/{num_pairs} pairs")

    # Save annotations as both CSV and JSON
    import pandas as pd

    df = pd.DataFrame(pairs)
    csv_path = output_dir / "pairs.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV annotations to {csv_path}")

    json_path = output_dir / "pairs.json"
    with open(json_path, "w") as f:
        json.dump(pairs, f, indent=2)
    logger.info(f"Saved JSON annotations to {json_path}")

    # Statistics
    pos_count = sum(1 for p in pairs if p["label"] == "pos")
    logger.info(
        f"NDEC data generation complete: {num_pairs} pairs "
        f"({pos_count} positive, {num_pairs - pos_count} negative)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate dummy data for Adaptive-DINO-ICD testing"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--num_disc",
        type=int,
        default=100,
        help="Number of DISC images to generate",
    )
    parser.add_argument(
        "--num_ndec",
        type=int,
        default=50,
        help="Number of NDEC pairs to generate",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Generated image size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    image_size = (args.image_size, args.image_size)

    logger.info(f"Generating dummy data in {output_dir}")

    # Generate DISC data
    disc_dir = output_dir / "disc"
    generate_disc_data(disc_dir, args.num_disc, image_size)

    # Generate NDEC data
    ndec_dir = output_dir / "ndec"
    generate_ndec_data(ndec_dir, args.num_ndec, image_size=image_size)

    logger.info("=" * 50)
    logger.info("Dummy data generation complete!")
    logger.info(f"DISC images: {disc_dir}")
    logger.info(f"NDEC pairs: {ndec_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
