//! License plate recognizer using ONNX Runtime.

use crate::config::PlateConfig;
use crate::error::Result;
use crate::postprocessor::{decode_plate_output, decode_region_output};
use crate::preprocessor::preprocess_image;
use crate::types::PlatePrediction;
use ort::session::{Session, SessionOutputs};
use ort::value::TensorRef;
use std::path::Path;

/// License plate recognizer using ONNX Runtime.
///
/// This is the main API for performing OCR on license plate images.
pub struct LicensePlateRecognizer {
    /// ONNX Runtime session for inference.
    session: Session,

    /// Configuration for the model.
    config: PlateConfig,

    /// Name of the plate output tensor.
    plate_output_name: String,

    /// Name of the region output tensor (if available).
    region_output_name: Option<String>,

    /// Whether the model has region recognition support.
    has_region_head: bool,

    /// Name of the input tensor.
    input_name: String,
}

impl LicensePlateRecognizer {
    /// Create a new recognizer from model and config paths.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file
    /// * `config` - Plate configuration loaded from YAML
    ///
    /// # Example
    ///
    /// ```no_run
    /// use fast_plate_ocr::{LicensePlateRecognizer, PlateConfig};
    /// #
    /// # fn main() -> fast_plate_ocr::error::Result<()> {
    /// let config = PlateConfig::from_yaml("config.yaml")?;
    /// let recognizer = LicensePlateRecognizer::new("model.onnx", &config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new<P: AsRef<Path>>(model_path: P, config: &PlateConfig) -> Result<Self> {
        let model_path: &Path = model_path.as_ref();

        if !model_path.exists() {
            return Err(crate::error::OcrError::FileNotFound(
                model_path.to_path_buf(),
            ));
        }

        // Create ONNX Runtime session
        let session = Session::builder()?.commit_from_file(model_path)?;

        // Get output names
        let outputs = session.outputs();
        let mut plate_output_name = String::new();
        let mut region_output_name = None;
        let mut input_name = String::from("input"); // default

        // Get input name
        let inputs = session.inputs();
        for input in inputs.iter() {
            let name = input.name();
            input_name = name.to_string();
            break; // Use the first input
        }

        for output in outputs.iter() {
            let name = output.name();
            if name.contains("plate") {
                plate_output_name = name.to_string();
            } else if name.contains("region") {
                region_output_name = Some(name.to_string());
            }
        }

        // If no "plate" output found, use the first output
        if plate_output_name.is_empty() {
            if let Some(first_output) = outputs.first() {
                plate_output_name = first_output.name().to_string();
            }
        }

        let has_region_head = region_output_name.is_some()
            && config.has_region_recognition();

        Ok(Self {
            session,
            config: config.clone(),
            plate_output_name,
            region_output_name,
            has_region_head,
            input_name,
        })
    }

    /// Run inference on a single image.
    ///
    /// # Arguments
    ///
    /// * `image_path` - Path to the image file
    ///
    /// # Returns
    ///
    /// A single plate prediction.
    pub fn run<P: AsRef<Path>>(&mut self, image_path: P) -> Result<PlatePrediction> {
        let mut results = self.run_impl(&[image_path.as_ref().to_path_buf()])?;
        if results.len() != 1 {
            return Err(crate::error::OcrError::InvalidShape(
                "Expected exactly 1 result".to_string(),
            ));
        }
        Ok(results.remove(0))
    }

    /// Run inference on multiple images.
    ///
    /// # Arguments
    ///
    /// * `image_paths` - Slice of paths to image files
    ///
    /// # Returns
    ///
    /// A vector of plate predictions, one per input image.
    pub fn run_batch<P: AsRef<Path>>(&mut self, image_paths: &[P]) -> Result<Vec<PlatePrediction>> {
        let paths: Vec<std::path::PathBuf> = image_paths
            .iter()
            .map(|p| p.as_ref().to_path_buf())
            .collect();
        self.run_impl(&paths)
    }

    /// Internal implementation of inference.
    fn run_impl(&mut self, image_paths: &[std::path::PathBuf]) -> Result<Vec<PlatePrediction>> {
        // Preprocess all images
        let mut all_tensors = Vec::new();
        for path in image_paths {
            let img = image::open(path)?;
            let tensor = preprocess_image(&img, &self.config)?;
            all_tensors.push(tensor);
        }

        // Stack into batch
        let batch_size = all_tensors.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        // For simplicity, process one at a time if batch_size > 1
        // TODO: Implement proper batching
        let mut results = Vec::with_capacity(batch_size);

        for tensor in all_tensors {
            let result = self.run_single(tensor)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Run inference on a single preprocessed tensor.
    fn run_single(&mut self, tensor: ndarray::Array4<u8>) -> Result<PlatePrediction> {
        println!("Input tensor shape: {:?}", tensor.shape());

        // Create input tensor reference from ndarray
        let input_tensor = TensorRef::from_array_view(&tensor)?;

        // Run inference
        println!("Running inference...");
        let inputs = ort::inputs![self.input_name.as_str() => input_tensor];
        let outputs: SessionOutputs = self.session.run(inputs)?;

        // Extract plate output directly using indexing (like yolov8 example)
        println!("Extracting plate output...");
        let plate_extracted = outputs[self.plate_output_name.as_str()].try_extract_array::<f32>()?;

        println!("Plate output shape: {:?}", plate_extracted.shape());

        let plate_array = plate_extracted.view().into_dyn();

        let mut predictions = decode_plate_output(
            &plate_array,
            self.config.max_plate_slots,
            &self.config.alphabet,
            Some(self.config.pad_char),
            true,
            self.has_region_head,
        )?;

        if predictions.is_empty() {
            return Ok(PlatePrediction::new(""));
        }

        let mut prediction = predictions.remove(0);

        // Decode region if available
        if self.has_region_head {
            if let Some(region_name) = &self.region_output_name {
                let region_extracted = outputs[region_name.as_str()].try_extract_array::<f32>()?;
                let region_array = region_extracted.view().into_dyn();
                if let Some(regions) = &self.config.plate_regions {
                    if let Ok(region_results) = decode_region_output(&region_array, regions) {
                        if !region_results.is_empty() {
                            prediction.region = Some(region_results[0].0.clone());
                            prediction.region_prob = Some(region_results[0].1);
                        }
                    }
                }
            }
        }

        Ok(prediction)
    }

    /// Get the configuration used by this recognizer.
    pub fn config(&self) -> &PlateConfig {
        &self.config
    }

    /// Get the model's output names.
    pub fn output_names(&self) -> Vec<String> {
        self.session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ColorMode, InterpolationMethod, PaddingColor};

    #[test]
    fn test_recognizer_creation() {
        let config = PlateConfig {
            max_plate_slots: 9,
            alphabet: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_".to_string(),
            pad_char: '_',
            img_height: 64,
            img_width: 128,
            keep_aspect_ratio: false,
            interpolation: InterpolationMethod::Linear,
            image_color_mode: ColorMode::Rgb,
            padding_color: PaddingColor::Gray(114),
            plate_regions: None,
        };

        // This will fail if model doesn't exist, but we're just testing compilation
        // In a real test, we'd need a test model
        let result = LicensePlateRecognizer::new("nonexistent.onnx", &config);
        assert!(result.is_err());
    }
}
