# fast-plate-ocr-rust

Rust port of the [fast-plate-ocr](https://github.com/andreader Silva/fast-plate-ocr) library for license plate OCR using ONNX Runtime.

## Features

- 🚀 Fast and memory-safe Rust implementation
- 📦 ONNX Runtime for model inference
- 🌍 Multi-country plate recognition (65+ countries)
- 🎯 Region detection support
- 💻 Easy-to-use CLI and library API

## Prerequisites

- Rust 1.70+ (install from [rustup.rs](https://rustup.rs))
- ONNX model file and config YAML

## Installation

### Build from source

```bash
git clone <repository-url>
cd fast-plate-ocr-rust
cargo build --release
```

The binary will be available at `target/release/fast_plate_ocr`.

## Usage

### Command Line

Basic usage:
```bash
fast_plate_ocr <model.onnx> <config.yaml> <image.jpg>
```

Example:
```bash
fast_plate_ocr cct_xs_v2_global.onnx cct_xs_v2_global_plate_config.yaml plate.jpg
```

Output:
```
Plate: ABC1234
Region: United States
```

### Multiple images

```bash
fast_plate_ocr model.onnx config.yaml image1.jpg image2.jpg image3.jpg
```

### As a Library

Add to your `Cargo.toml`:
```toml
[dependencies]
fast_plate_ocr = { path = "/path/to/fast-plate-ocr-rust" }
```

Example code:
```rust
use fast_plate_ocr::{LicensePlateRecognizer, PlateConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = PlateConfig::from_yaml("config.yaml")?;

    // Create recognizer
    let mut recognizer = LicensePlateRecognizer::new("model.onnx", &config)?;

    // Run inference on a single image
    let result = recognizer.run("plate.jpg")?;
    println!("Plate: {}", result.plate);
    if let Some(region) = result.region {
        println!("Region: {}", region);
    }

    // Or run on multiple images
    let results = recognizer.run_batch(&["image1.jpg", "image2.jpg"])?;
    for result in results {
        println!("Plate: {}", result.plate);
    }

    Ok(())
}
```

## Configuration

The config YAML file specifies model parameters:

```yaml
max_plate_slots: 10
alphabet: '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
pad_char: '_'
img_height: 64
img_width: 128
keep_aspect_ratio: false
interpolation: linear
image_color_mode: rgb
plate_regions:
  - United States
  - Europe
  - Argentina
  # ... more regions
```

## Model Requirements

**Important:** This model requires **cropped license plate images**, not full traffic scenes.

For full traffic scene images, you need a two-stage pipeline:
1. **License plate detector** (e.g., YOLO) to detect and crop plates
2. **OCR model** (this library) to read the text

### Input Image Requirements

- Format: JPEG, PNG, or other image formats supported by the `image` crate
- Content: Cropped license plate (not full scene)
- Recommended size: Models typically resize to 128x64 internally

## Downloading Models

Models can be downloaded from the original [fast-plate-ocr](https://github.com/andreadier silva/fast-plate-ocr) project or use the Python version to download:

```bash
pip install fast-plate-ocr
# Models will be downloaded to ~/.cache/fast-plate-ocr/
```

Copy the `.onnx` model and corresponding `.yaml` config to your project.

## API Reference

### `PlateConfig`

Configuration loaded from YAML file.

```rust
pub struct PlateConfig {
    pub max_plate_slots: usize,
    pub alphabet: String,
    pub pad_char: char,
    pub img_height: u32,
    pub img_width: u32,
    pub keep_aspect_ratio: bool,
    pub interpolation: InterpolationMethod,
    pub image_color_mode: ColorMode,
    pub plate_regions: Option<Vec<String>>,
}
```

### `LicensePlateRecognizer`

Main inference class.

```rust
impl LicensePlateRecognizer {
    pub fn new<P: AsRef<Path>>(model_path: P, config: &PlateConfig) -> Result<Self>
    pub fn run<P: AsRef<Path>>(&mut self, image_path: P) -> Result<PlatePrediction>
    pub fn run_batch<P: AsRef<Path>>(&mut self, image_paths: &[P]) -> Result<Vec<PlatePrediction>>
}
```

### `PlatePrediction`

Prediction result.

```rust
pub struct PlatePrediction {
    pub plate: String,
    pub char_probs: Option<Vec<f32>>,
    pub region: Option<String>,
    pub region_prob: Option<f32>,
}
```

## Performance

Compared to the Python implementation:
- **Lower latency** due to native compilation
- **Reduced memory footprint**
- **No runtime dependencies** (standalone binary)
- **Same accuracy** as Python reference implementation

## Development

Run tests:
```bash
cargo test

Run with logging:
```bash
RUST_LOG=debug cargo run -- model.onnx config.yaml image.jpg
```

## License

See LICENSE file for details.

## Acknowledgments

- Original [fast-plate-ocr](https://github.com/andreadier Silva/fast-plate-ocr) Python library
- ONNX Runtime for the inference engine
- All contributors to the dependent libraries
