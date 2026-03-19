//! Automatic License Plate Recognition (ALPR) system.
//!
//! Combines license plate detection with OCR reading.

use crate::detector::{DetectionResult, LicensePlateDetector};
use crate::error::Result;
use crate::recognizer::LicensePlateRecognizer;
use image::DynamicImage;
use std::path::Path;

/// Complete ALPR result including detection and OCR.
#[derive(Debug, Clone)]
pub struct ALPRResult {
    /// Detection result (bounding box and confidence).
    pub detection: DetectionResult,
    /// OCR result (plate text and region).
    pub ocr: Option<OCRResult>,
}

/// OCR result for a detected license plate.
#[derive(Debug, Clone)]
pub struct OCRResult {
    /// Recognized plate text.
    pub plate: String,
    /// Region/country prediction (if available).
    pub region: Option<String>,
    /// Per-character confidence scores (if available).
    pub char_probs: Option<Vec<f32>>,
}

/// Automatic License Plate Recognition system.
///
/// Combines a YOLO-v9 detector with fast-plate-ocr for complete
/// license plate recognition from full traffic scene images.
pub struct ALPR {
    detector: LicensePlateDetector,
    ocr: LicensePlateRecognizer,
}

impl ALPR {
    /// Create a new ALPR system.
    ///
    /// # Arguments
    ///
    /// * `detector_model_path` - Path to YOLO-v9 ONNX model
    /// * `detector_conf_thresh` - Confidence threshold for detection (0.0-1.0)
    /// * `ocr_model_path` - Path to OCR ONNX model
    /// * `ocr_config` - OCR configuration
    pub fn new<P: AsRef<Path>>(
        detector_model_path: P,
        detector_conf_thresh: f32,
        ocr_model_path: P,
        ocr_config: &crate::PlateConfig,
    ) -> Result<Self> {
        let detector = LicensePlateDetector::new(detector_model_path, detector_conf_thresh)?;
        let ocr = LicensePlateRecognizer::new(ocr_model_path, ocr_config)?;

        Ok(Self { detector, ocr })
    }

    /// Run ALPR on an image file.
    ///
    /// # Arguments
    ///
    /// * `image_path` - Path to the image file
    ///
    /// # Returns
    ///
    /// A vector of ALPR results, one for each detected license plate.
    pub fn predict<P: AsRef<Path>>(&mut self, image_path: P) -> Result<Vec<ALPRResult>> {
        let image_path = image_path.as_ref();

        // Load image
        let img = image::open(image_path)?;

        // Run detection
        let detections = self.detector.predict(image_path)?;

        println!("Found {} license plate(s)", detections.len());

        // Run OCR on each detected plate
        let mut results = Vec::new();
        for detection in detections {
            let bbox = &detection.bounding_box;

            // Crop the detected plate
            let cropped = self.crop_plate(&img, bbox)?;

            // Run OCR on the cropped plate
            let ocr_result = self.run_ocr_on_crop(cropped)?;

            results.push(ALPRResult {
                detection,
                ocr: ocr_result,
            });
        }

        Ok(results)
    }

    /// Crop a license plate from an image using the bounding box.
    fn crop_plate(&self, img: &DynamicImage, bbox: &crate::detector::BoundingBox) -> Result<DynamicImage> {
        let img_width = img.width() as i32;
        let img_height = img.height() as i32;

        // Clamp coordinates to image bounds
        let x1 = bbox.x1.max(0).min(img_width - 1) as u32;
        let y1 = bbox.y1.max(0).min(img_height - 1) as u32;
        let x2 = bbox.x2.max(0).min(img_width).min(img_width) as u32;
        let y2 = bbox.y2.max(0).min(img_height).min(img_height) as u32;

        // Ensure valid crop dimensions
        if x2 <= x1 || y2 <= y1 {
            return Err(crate::error::OcrError::InvalidShape(
                "Invalid bounding box coordinates".to_string(),
            ));
        }

        let width = x2 - x1;
        let height = y2 - y1;

        // Crop the image
        Ok(img.crop_imm(x1, y1, width, height))
    }

    /// Run OCR on a cropped license plate image.
    fn run_ocr_on_crop(&mut self, crop: DynamicImage) -> Result<Option<OCRResult>> {
        // Save crop to temp file and run OCR
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("temp_plate_crop.png");

        // Save cropped image
        crop.save(&temp_file)?;

        // Run OCR
        let prediction = self.ocr.run(&temp_file)?;

        if prediction.plate.is_empty() {
            Ok(None)
        } else {
            Ok(Some(OCRResult {
                plate: prediction.plate,
                region: prediction.region,
                char_probs: prediction.char_probs,
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpr_creation() {
        use crate::PlateConfig;

        // This will fail if models don't exist
        let config = PlateConfig {
            max_plate_slots: 10,
            alphabet: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_".to_string(),
            pad_char: '_',
            img_height: 64,
            img_width: 128,
            keep_aspect_ratio: false,
            interpolation: crate::InterpolationMethod::Linear,
            image_color_mode: crate::ColorMode::Rgb,
            padding_color: crate::PaddingColor::Gray(114),
            plate_regions: None,
        };

        let result = ALPR::new(
            "nonexistent_detector.onnx",
            0.4,
            "nonexistent_ocr.onnx",
            &config,
        );
        assert!(result.is_err());
    }
}
