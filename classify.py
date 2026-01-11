#!/usr/bin/env python3
"""
NSFW Image Classifier CLI

Classifies images from input directories into "normal" and "nsfw" categories
using the Falconsai/nsfw_image_detection model from Hugging Face.
"""

import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    global Image, pipeline, torch

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

    try:
        import torch
    except ImportError:
        print("Error: torch is not installed. Run: pip install torch", file=sys.stderr)
        sys.exit(1)

    # Register HEIF/HEIC format support if available
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass

    return Image, pipeline, torch


def get_device(torch):
    """Detect the best available device (MPS for Apple Silicon, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_image(Image, image_path: Path):
    """Load and preprocess a single image. Returns (path, image) or (path, None) on error."""
    try:
        img = Image.open(image_path)
        # Convert to RGB if necessary (for RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return (image_path, img)
    except Exception as e:
        print(f"Error loading {image_path}: {e}", file=sys.stderr)
        return (image_path, None)


def classify_batch(classifier, images: list) -> list:
    """Classify a batch of images. Returns list of labels."""
    try:
        results = classifier(images)
        # Results is a list of lists, one per image
        labels = []
        for result in results:
            top_result = max(result, key=lambda x: x['score'])
            labels.append(top_result['label'].lower())
        return labels
    except Exception as e:
        print(f"Error during batch classification: {e}", file=sys.stderr)
        return [None] * len(images)


def copy_file(src: Path, dest: Path) -> bool:
    """Copy a file to destination. Returns True on success."""
    try:
        shutil.copy2(src, dest)
        return True
    except Exception as e:
        print(f"Error copying {src}: {e}", file=sys.stderr)
        return False


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

    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=8,
        help='Number of images to process in each batch (default: 8)'
    )

    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of worker threads for I/O operations (default: 4)'
    )

    args = parser.parse_args()

    # Load dependencies after argument parsing (so --help works without deps)
    Image, pipeline, torch = load_dependencies()

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

    # Detect best available device
    device = get_device(torch)
    print(f"Using device: {device}")

    # Load the classifier
    print(f"Loading model: {args.model}...")
    classifier = pipeline("image-classification", model=args.model, device=device)
    print("Model loaded successfully")
    print(f"Batch size: {args.batch_size}, Workers: {args.workers}")

    # Process images in batches with parallel I/O
    stats = {'normal': 0, 'nsfw': 0, 'errors': 0}
    total_images = len(images)
    processed = 0

    # Process in batches
    for batch_start in range(0, total_images, args.batch_size):
        batch_end = min(batch_start + args.batch_size, total_images)
        batch_paths = images[batch_start:batch_end]

        # Parallel image loading
        loaded_images = []
        valid_paths = []

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(load_image, Image, path): path for path in batch_paths}
            for future in as_completed(futures):
                path, img = future.result()
                if img is not None:
                    loaded_images.append(img)
                    valid_paths.append(path)
                else:
                    stats['errors'] += 1
                    processed += 1
                    print(f"[{processed}/{total_images}] {path.name}: ERROR (load failed)")

        if not loaded_images:
            continue

        # Batch classification
        print(f"[{processed + 1}-{processed + len(loaded_images)}/{total_images}] Classifying batch of {len(loaded_images)} images...", end=' ', flush=True)
        labels = classify_batch(classifier, loaded_images)
        print("done")

        # Prepare copy operations
        copy_tasks = []
        for path, label in zip(valid_paths, labels):
            processed += 1

            if label is None:
                stats['errors'] += 1
                print(f"  {path.name}: ERROR (classification failed)")
                continue

            # Determine destination
            if label == 'nsfw':
                dest_dir = nsfw_dir
            else:
                dest_dir = normal_dir
                label = 'normal'

            # Handle filename conflicts
            dest_path = dest_dir / path.name
            if dest_path.exists():
                stem = path.stem
                suffix = path.suffix
                counter = 1
                while dest_path.exists():
                    dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

            copy_tasks.append((path, dest_path, label))

        # Parallel file copying
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(copy_file, src, dest): (src, dest, label)
                      for src, dest, label in copy_tasks}
            for future in as_completed(futures):
                src, dest, label = futures[future]
                if future.result():
                    stats[label] += 1
                    print(f"  {src.name}: {label.upper()}")
                else:
                    stats['errors'] += 1

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
