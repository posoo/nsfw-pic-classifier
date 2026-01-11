# NSFW Image Classifier

A CLI tool that classifies images into "normal" and "nsfw" categories using the [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection) model from Hugging Face.

## Installation

```bash
uv sync
```

## Usage

```bash
# Basic usage
uv run python classify.py --input ./photos

# With output directory and recursive search
uv run python classify.py --input ./photos --output ./classified --recursive

# Multiple input directories
uv run python classify.py -i ./dir1 -i ./dir2 -r -o ./output
```

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--input` | `-i` | Input directory containing images (required, can be specified multiple times) |
| `--output` | `-o` | Output directory for classified images (default: current directory) |
| `--recursive` | `-r` | Recursively search for images in subdirectories |
| `--model` | | Custom Hugging Face model (default: `Falconsai/nsfw_image_detection`) |

## Supported Image Formats

`.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`, `.tiff`, `.tif`, `.heif`, `.heic`, `.avif`

## Output

Images are **copied** (not moved) into `normal/` and `nsfw/` folders under the output directory.
