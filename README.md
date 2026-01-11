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
| `--batch-size` | `-b` | Number of images to process in each batch (default: 8) |
| `--workers` | `-w` | Number of worker threads for I/O operations (default: 4) |

## Performance

The classifier uses parallel processing for optimal performance:

- **GPU Acceleration**: Automatically detects and uses MPS (Apple Silicon), CUDA, or CPU
- **Batch Inference**: Processes multiple images in a single model forward pass
- **Parallel I/O**: Loads and copies files using multiple threads

### Tuning Tips

```bash
# Larger batches for GPUs with more VRAM
uv run python classify.py -i ./photos -b 16 -w 8

# Smaller batches if running out of memory
uv run python classify.py -i ./photos -b 4

# More workers for network storage or fast NVMe
uv run python classify.py -i ./photos -w 16
```

## Supported Image Formats

`.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`, `.tiff`, `.tif`, `.heif`, `.heic`, `.avif`

## Output

Images are **copied** (not moved) into `normal/` and `nsfw/` folders under the output directory.
