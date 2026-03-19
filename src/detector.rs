//! License plate detection using YOLO-v9 ONNX model.

use crate::error::Result;
use ndarray::Array4;
use ort::session::{Session, SessionOutputs};
use ort::value::TensorRef;
use std::path::Path;

/// Bounding box for a detected license plate.
#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

/// Detection result from the license plate detector.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    pub label: String,
    pub confidence: f32,
    pub bounding_box: BoundingBox,
}

/// License plate detector using YOLO-v9 ONNX model.
pub struct LicensePlateDetector {
    /// ONNX Runtime session for inference.
    session: Session,

    /// Name of the input tensor.
    input_name: String,

    /// Name of the output tensor.
    output_name: String,

    /// Input image size (assumed square).
    img_size: usize,

    /// Confidence threshold for detections.
    conf_thresh: f32,

    /// Class labels for the detector.
    class_labels: Vec<String>,
}

/// Parse image input size from YOLO model filename (e.g. "yolo-v9-t-384-..." → 384).
fn parse_img_size_from_path(model_path: &Path) -> usize {
    let filename = model_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    for part in filename.split('-') {
        if let Ok(n) = part.parse::<usize>() {
            if (128..=1024).contains(&n) {
                return n;
            }
        }
    }
    384
}

impl LicensePlateDetector {
    /// Create a new detector from a model file.
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        conf_thresh: f32,
    ) -> Result<Self> {
        let model_path: &Path = model_path.as_ref();

        if !model_path.exists() {
            return Err(crate::error::OcrError::FileNotFound(
                model_path.to_path_buf(),
            ));
        }

        // Create ONNX Runtime session
        let session = Session::builder()?.commit_from_file(model_path)?;

        // Get input and output names
        let inputs = session.inputs();
        let mut input_name = String::new();

        for input in inputs.iter() {
            input_name = input.name().to_string();
            break; // Use the first input
        }

        let outputs = session.outputs();
        let mut output_name = String::new();

        for output in outputs.iter() {
            output_name = output.name().to_string();
            break; // Use the first output
        }

        // Parse image size from model filename (e.g. yolo-v9-t-384-... → 384)
        let img_size = parse_img_size_from_path(model_path);

        // Default class label for license plate
        let class_labels = vec!["license_plate".to_string()];

        Ok(Self {
            session,
            input_name,
            output_name,
            img_size,
            conf_thresh,
            class_labels,
        })
    }

    /// Run detection on an image.
    pub fn predict(&mut self, image_path: &Path) -> Result<Vec<DetectionResult>> {
        // Load and preprocess image
        let img = image::open(image_path)?;

        // Preprocess for YOLOv9
        let (preprocessed, ratio, (dw, dh)) = self.preprocess_yolo(&img)?;

        println!("Detector input shape: {:?}", preprocessed.shape());

        // Create input tensor reference
        let input_tensor = TensorRef::from_array_view(&preprocessed)?;

        // Run inference
        let inputs = ort::inputs![self.input_name.as_str() => input_tensor];
        let outputs: SessionOutputs = self.session.run(inputs)?;

        // Extract output
        let output_extracted = outputs[self.output_name.as_str()].try_extract_array::<f32>()?;
        let output_view = output_extracted.view().into_dyn();

        println!("Detector output shape: {:?}", output_view.shape());

        // Parse detections - we need to copy data before SessionOutputs is dropped
        let shape = output_view.shape().to_vec();
        let mut raw_data = Vec::new();

        // Copy data we need before SessionOutputs is dropped
        if shape.len() >= 2 && !shape.is_empty() {
            let num_detections = shape[0];
            if num_detections > 0 {
                // Collect the data as a flat vector first
                let _total_elements = output_view.len();
                let flat_data: Vec<f32> = output_view.iter().copied().collect();

                // Reshape into rows of 7 columns
                if flat_data.len() >= num_detections * 7 {
                    for i in 0..num_detections {
                        let mut row = [0f32; 7];
                        for j in 0..7 {
                            row[j] = flat_data[i * shape.get(1).unwrap_or(&7) + j];
                        }
                        raw_data.push(row);
                    }
                }
            }
        }

        // Drop outputs explicitly to release the mutable borrow
        drop(outputs);

        // Now we can call postprocess with the copied data
        let detections = self.postprocess_with_raw_data(raw_data, ratio, dw, dh)?;

        Ok(detections)
    }

    /// Preprocess image for YOLOv9 detection (letterbox + normalize).
    fn preprocess_yolo(
        &self,
        img: &image::DynamicImage,
    ) -> Result<(Array4<f32>, (f32, f32), (f32, f32))> {
        let (orig_h, orig_w) = (img.height() as f32, img.width() as f32);
        let target_size = self.img_size as f32;

        // Calculate scaling ratio
        let r = (target_size / orig_h).min(target_size / orig_w);
        let new_unpad_w = (orig_w * r).round();
        let new_unpad_h = (orig_h * r).round();

        // Resize image (use resize_exact to avoid aspect ratio recomputation)
        let resized = img.resize_exact(
            new_unpad_w as u32,
            new_unpad_h as u32,
            image::imageops::FilterType::Triangle,
        );

        // Calculate padding
        let dw = (target_size - new_unpad_w) / 2.0;
        let dh = (target_size - new_unpad_h) / 2.0;

        let top = dh.floor() as u32;
        let bottom = dh.ceil() as u32;
        let left = dw.floor() as u32;
        let right = dw.ceil() as u32;

        // Apply letterbox padding
        let padded = self.letterbox(&resized, self.img_size as u32, top, bottom, left, right);

        // Convert to RGB and normalize
        let rgb_img = padded.to_rgb8();
        let h = self.img_size;
        let w = self.img_size;

        // Create CHW format directly to avoid non-contiguous arrays
        // CHW format: [batch, channels, height, width]
        let mut pixels = vec![0f32; 1 * 3 * h * w];

        for y in 0..h {
            for x in 0..w {
                let pixel = rgb_img.get_pixel(x as u32, y as u32);
                let idx_base = y * w + x;

                // R channel
                pixels[0 * h * w + idx_base] = pixel[0] as f32 / 255.0;
                // G channel
                pixels[1 * h * w + idx_base] = pixel[1] as f32 / 255.0;
                // B channel
                pixels[2 * h * w + idx_base] = pixel[2] as f32 / 255.0;
            }
        }

        // Create tensor with shape (1, 3, H, W) - CHW format for YOLO
        let chw_data = Array4::from_shape_vec((1, 3, h, w), pixels)?;

        Ok((chw_data, (r, r), (dw, dh)))
    }

    /// Apply letterbox padding to an image.
    fn letterbox(
        &self,
        img: &image::DynamicImage,
        _target_size: u32,
        top: u32,
        bottom: u32,
        left: u32,
        right: u32,
    ) -> image::DynamicImage {
        let rgb_img = img.to_rgb8();
        let padded_w = rgb_img.width() + left + right;
        let padded_h = rgb_img.height() + top + bottom;

        let mut padded = image::RgbImage::new(padded_w, padded_h);
        let fill = image::Rgb([114, 114, 114]); // Gray padding color

        // Fill with padding color
        for y in 0..padded_h {
            for x in 0..padded_w {
                padded.put_pixel(x, y, fill);
            }
        }

        // Copy original image
        for y in 0..rgb_img.height() {
            for x in 0..rgb_img.width() {
                let pixel = rgb_img.get_pixel(x, y);
                padded.put_pixel(x + left, y + top, *pixel);
            }
        }

        image::DynamicImage::ImageRgb8(padded)
    }

    /// Postprocess raw detection outputs into detection results.
    #[allow(dead_code)]
    fn postprocess_detections(
        &self,
        output: ndarray::ArrayViewD<f32>,
        ratio: (f32, f32),
        dw: f32,
        dh: f32,
    ) -> Result<Vec<DetectionResult>> {
        let mut results = Vec::new();

        // YOLOv9 output format: [N, 7] where columns are:
        // [batch_idx, x1, y1, x2, y2, class_id, confidence]
        let shape = output.shape();

        if shape.len() < 2 {
            return Ok(results);
        }

        let num_detections = shape[0];
        if num_detections == 0 {
            return Ok(results);
        }

        for i in 0..num_detections {
            // Get confidence (last column)
            let confidence = output[[i, 6]];

            // Filter by confidence threshold
            if confidence < self.conf_thresh {
                continue;
            }

            // Get bounding box coordinates
            let x1 = output[[i, 1]];
            let y1 = output[[i, 2]];
            let x2 = output[[i, 3]];
            let y2 = output[[i, 4]];

            // Get class ID
            let class_id = output[[i, 5]] as i32;

            // Convert coordinates back to original image size
            let x1_orig = ((x1 - dw) / ratio.0).round() as i32;
            let y1_orig = ((y1 - dh) / ratio.1).round() as i32;
            let x2_orig = ((x2 - dw) / ratio.0).round() as i32;
            let y2_orig = ((y2 - dh) / ratio.1).round() as i32;

            // Get class label
            let label = self
                .class_labels
                .get(class_id as usize)
                .cloned()
                .unwrap_or(format!("class_{}", class_id));

            results.push(DetectionResult {
                label,
                confidence,
                bounding_box: BoundingBox {
                    x1: x1_orig,
                    y1: y1_orig,
                    x2: x2_orig,
                    y2: y2_orig,
                },
            });
        }

        Ok(results)
    }

    /// Postprocess detections from raw data array (avoids lifetime issues).
    fn postprocess_with_raw_data(
        &self,
        raw_data: Vec<[f32; 7]>,
        ratio: (f32, f32),
        dw: f32,
        dh: f32,
    ) -> Result<Vec<DetectionResult>> {
        let mut results = Vec::new();

        // raw_data format: [batch_idx, x1, y1, x2, y2, class_id, confidence]
        for row in raw_data {
            let confidence = row[6];

            // Filter by confidence threshold
            if confidence < self.conf_thresh {
                continue;
            }

            // Get bounding box coordinates
            let x1 = row[1];
            let y1 = row[2];
            let x2 = row[3];
            let y2 = row[4];

            // Get class ID
            let class_id = row[5] as i32;

            // Convert coordinates back to original image size
            let x1_orig = ((x1 - dw) / ratio.0).round() as i32;
            let y1_orig = ((y1 - dh) / ratio.1).round() as i32;
            let x2_orig = ((x2 - dw) / ratio.0).round() as i32;
            let y2_orig = ((y2 - dh) / ratio.1).round() as i32;

            // Get class label
            let label = self
                .class_labels
                .get(class_id as usize)
                .cloned()
                .unwrap_or(format!("class_{}", class_id));

            results.push(DetectionResult {
                label,
                confidence,
                bounding_box: BoundingBox {
                    x1: x1_orig,
                    y1: y1_orig,
                    x2: x2_orig,
                    y2: y2_orig,
                },
            });
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_creation() {
        let result = LicensePlateDetector::new("nonexistent.onnx", 0.4);
        assert!(result.is_err());
    }
}
