#!/usr/bin/env python3
"""
NSFW Image Classifier CLI

Classifies images from input directories into "normal" and "nsfw" categories
using the Falconsai/nsfw_image_detection model from Hugging Face.
"""

import argparse
import shutil
import sys
from pathlib import Path

# Supported image extensions
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp',
    '.tiff', '.tif', '.heif', '.heic', '.avif'
}


def find_images(input_dirs: list[Path], recursive: bool) -> list[Path]:
    """Find all image files in the given directories."""
    images = []

    for input_dir in input_dirs:
        if not input_dir.exists():
            print(f"Warning: Input directory does not exist: {input_dir}", file=sys.stderr)
            continue

        if not input_dir.is_dir():
            print(f"Warning: Not a directory: {input_dir}", file=sys.stderr)
            continue

        if recursive:
            pattern = '**/*'
        else:
            pattern = '*'

        for file_path in input_dir.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(file_path)

    return images


def load_dependencies():
    """Load heavy dependencies lazily."""
    global Image, pipeline

    try:
        from PIL import Image
    except ImportError:
        print("Error: Pillow is not installed. Run: pip install pillow", file=sys.stderr)
        sys.exit(1)

    try:
        from transformers import pipeline
    except ImportError:
        print("Error: transformers is not installed. Run: pip install transformers torch", file=sys.stderr)
        sys.exit(1)

    # Register HEIF/HEIC format support if available
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass

    return Image, pipeline


def classify_image(Image, classifier, image_path: Path) -> str:
    """Classify a single image as 'normal' or 'nsfw'."""
    try:
        img = Image.open(image_path)
        # Convert to RGB if necessary (for RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        results = classifier(img)
        # Get the label with highest score
        top_result = max(results, key=lambda x: x['score'])
        return top_result['label'].lower()
    except Exception as e:
        print(f"Error processing {image_path}: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Classify images into "normal" and "nsfw" categories.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input ./photos
  %(prog)s --input ./photos --output ./classified --recursive
  %(prog)s --input ./dir1 --input ./dir2 -r
        """
    )

    parser.add_argument(
        '--input', '-i',
        action='append',
        required=True,
        type=Path,
        dest='input_dirs',
        help='Input directory containing images (can be specified multiple times)'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('.'),
        help='Output directory for classified images (default: current directory)'
    )

    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Recursively search for images in subdirectories'
    )

    parser.add_argument(
        '--model',
        default='Falconsai/nsfw_image_detection',
        help='Hugging Face model to use (default: Falconsai/nsfw_image_detection)'
    )

    args = parser.parse_args()

    # Load dependencies after argument parsing (so --help works without deps)
    Image, pipeline = load_dependencies()

    # Find all images
    print(f"Searching for images in {len(args.input_dirs)} director{'y' if len(args.input_dirs) == 1 else 'ies'}...")
    images = find_images(args.input_dirs, args.recursive)

    if not images:
        print("No images found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(images)} image(s)")

    # Create output directories
    normal_dir = args.output / 'normal'
    nsfw_dir = args.output / 'nsfw'

    normal_dir.mkdir(parents=True, exist_ok=True)
    nsfw_dir.mkdir(parents=True, exist_ok=True)

    # Load the classifier
    print(f"Loading model: {args.model}...")
    classifier = pipeline("image-classification", model=args.model)
    print("Model loaded successfully")

    # Process each image
    stats = {'normal': 0, 'nsfw': 0, 'errors': 0}

    for i, image_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] Processing: {image_path.name}...", end=' ')

        label = classify_image(Image, classifier, image_path)

        if label is None:
            stats['errors'] += 1
            print("ERROR")
            continue

        # Determine destination
        if label == 'nsfw':
            dest_dir = nsfw_dir
        else:
            dest_dir = normal_dir
            label = 'normal'  # Normalize label

        # Handle filename conflicts
        dest_path = dest_dir / image_path.name
        if dest_path.exists():
            stem = image_path.stem
            suffix = image_path.suffix
            counter = 1
            while dest_path.exists():
                dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        # Copy the file
        shutil.copy2(image_path, dest_path)
        stats[label] += 1
        print(f"{label.upper()}")

    # Print summary
    print("\n" + "=" * 40)
    print("Classification complete!")
    print(f"  Normal: {stats['normal']}")
    print(f"  NSFW:   {stats['nsfw']}")
    if stats['errors'] > 0:
        print(f"  Errors: {stats['errors']}")
    print(f"\nOutput directories:")
    print(f"  Normal: {normal_dir.absolute()}")
    print(f"  NSFW:   {nsfw_dir.absolute()}")


if __name__ == '__main__':
    main()
