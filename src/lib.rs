//! Fast Plate OCR - Rust implementation of license plate recognition using ONNX Runtime.
//!
//! This library provides OCR functionality for license plate images using pre-trained ONNX models.
//!
//! # Example
//!
//! ```no_run
//! use fast_plate_ocr::{LicensePlateRecognizer, PlateConfig};
//!
//! # fn main() -> fast_plate_ocr::error::Result<()> {
//! // Load configuration
//! let config = PlateConfig::from_yaml("config/latin_plate_config.yaml")?;
//!
//! // Create recognizer
//! let recognizer = LicensePlateRecognizer::new(
//!     "models/model.onnx",
//!     &config
//! )?;
//!
//! // Run inference
//! let result = recognizer.run("test_plate.jpg")?;
//! println!("Plate: {}", result.plate);
//! # Ok(())
//! # }
//! ```

pub mod alpr;
pub mod config;
pub mod detector;
pub mod error;
pub mod postprocessor;
pub mod preprocessor;
pub mod recognizer;
pub mod types;

// Re-export commonly used types
pub use alpr::{ALPR, ALPRResult, OCRResult};
pub use config::PlateConfig;
pub use detector::{BoundingBox, DetectionResult, LicensePlateDetector};
pub use error::{OcrError, Result};
pub use postprocessor::{decode_plate_output, decode_region_output};
pub use recognizer::LicensePlateRecognizer;
pub use types::{ColorMode, InterpolationMethod, PaddingColor, PlatePrediction};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_structure() {
        // Basic test to ensure library structure is correct
        assert_eq!(ColorMode::Rgb.num_channels(), 3);
        assert_eq!(ColorMode::Grayscale.num_channels(), 1);
    }
}
