//! Postprocessing of model outputs into plate predictions.

use crate::error::Result;
use crate::types::PlatePrediction;
use ndarray::ArrayViewD;

/// Decode plate output from model inference.
///
/// # Arguments
///
/// * `model_output` - Raw output tensor with shape (N, max_plate_slots * vocab_size) or similar
/// * `max_plate_slots` - Maximum number of character positions
/// * `alphabet` - Character alphabet used by the model
/// * `pad_char` - Optional padding character to remove
/// * `remove_pad_char` - Whether to remove trailing padding characters
/// * `return_confidence` - Whether to include per-character confidence scores
///
/// # Returns
///
/// A vector of predictions, one per input sample.
pub fn decode_plate_output(
    model_output: &ArrayViewD<f32>,
    max_plate_slots: usize,
    alphabet: &str,
    pad_char: Option<char>,
    remove_pad_char: bool,
    return_confidence: bool,
) -> Result<Vec<PlatePrediction>> {
    let vocab_size = alphabet.chars().count();

    // Debug: print the output shape and dimensions
    println!("Model output ndim: {}", model_output.ndim());
    println!("Model output shape: {:?}", model_output.shape());
    println!("Model output len: {}", model_output.len());
    println!("Expected: max_plate_slots={}, vocab_size={}, total={}",
              max_plate_slots, vocab_size, max_plate_slots * vocab_size);

    // The output might be in different formats:
    // - (N, max_plate_slots, vocab_size)
    // - (N, max_plate_slots * vocab_size)
    // - (N, vocab_size, max_plate_slots)

    // Try to determine batch size
    let total_elements = model_output.len();
    let batch_size = if model_output.ndim() >= 2 {
        model_output.shape()[0]
    } else {
        1
    };

    println!("Detected batch_size: {}", batch_size);

    // Calculate expected size
    let expected_size = batch_size * max_plate_slots * vocab_size;
    if total_elements != expected_size {
        println!("WARNING: Size mismatch! got {}, expected {}", total_elements, expected_size);
        // Try to proceed anyway
    }

    // Convert to a 3D array with fixed dimensions
    let output_3d: ndarray::Array3<f32> = {
        let owned = model_output.to_owned();

        if model_output.ndim() == 1 {
            // 1D output - single sample, reshape
            println!("Reshaping 1D output to (1, {}, {})", max_plate_slots, vocab_size);
            owned.into_shape((1, max_plate_slots, vocab_size))?
        } else if model_output.ndim() == 2 {
            // 2D output
            let shape = model_output.shape();
            println!("2D output with shape {:?}", shape);
            if shape[0] == 1 && shape[1] == max_plate_slots * vocab_size {
                println!("Reshaping (1, {}) to (1, {}, {})", shape[1], max_plate_slots, vocab_size);
                owned.into_shape((1, max_plate_slots, vocab_size))?
            } else if shape[1] == max_plate_slots && shape[0] == vocab_size {
                // Transpose needed: (vocab_size, max_plate_slots) -> (max_plate_slots, vocab_size)
                println!("Transposing 2D array from ({}, {}) to (1, {}, {})",
                         shape[0], shape[1], max_plate_slots, vocab_size);
                let reshaped = owned.into_shape((vocab_size, max_plate_slots))?;
                let transposed = reshaped.reversed_axes();
                transposed.into_shape((1, max_plate_slots, vocab_size))?
            } else {
                println!("Trying to reshape 2D {:?} to (1, {}, {})", shape, max_plate_slots, vocab_size);
                owned.into_shape((1, max_plate_slots, vocab_size))?
            }
        } else if model_output.ndim() == 3 {
            // Already 3D
            let shape = model_output.shape();
            println!("3D output with shape {:?}", shape);
            if shape[1] == vocab_size && shape[2] == max_plate_slots {
                // Need to swap axes 1 and 2
                println!("Swapping axes from (N, vocab_size, max_plate_slots) to (N, max_plate_slots, vocab_size)");
                let output = owned.into_dimensionality::<ndarray::Ix3>()?;
                output.permuted_axes([0, 2, 1])
            } else {
                println!("Using 3D output as-is with shape {:?}", shape);
                owned.into_dimensionality::<ndarray::Ix3>()?
            }
        } else {
            return Err(crate::error::OcrError::InvalidShape(format!(
                "Unexpected output dimensions: {}", model_output.ndim()
            )));
        }
    };

    let alphabet_chars: Vec<char> = alphabet.chars().collect();

    // For dynamic array, we need to process based on actual dimensions
    let actual_batch_size = output_3d.shape()[0];
    let actual_slots = output_3d.shape()[1];
    let actual_vocab = output_3d.shape()[2];

    println!("Processing: batch_size={}, slots={}, vocab={}", actual_batch_size, actual_slots, actual_vocab);

    let mut predictions = Vec::with_capacity(actual_batch_size);

    for i in 0..actual_batch_size {
        let mut plate_text = String::new();
        let mut char_probs = if return_confidence {
            Some(Vec::with_capacity(actual_slots))
        } else {
            None
        };

        for j in 0..actual_slots {
            let slot_view = output_3d.slice(ndarray::s![i, j, ..]);
            let max_idx = slot_view
                .iter()
                .enumerate()
                .max_by(|a, b| {
                    a.1.partial_cmp(b.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            let max_prob = slot_view[max_idx];
            println!("Slot {}: idx={}, char={:?}, prob={:.4}",
                     j, max_idx,
                     alphabet_chars.get(max_idx).copied().unwrap_or('_'),
                     max_prob);

            if let Some(&ch) = alphabet_chars.get(max_idx) {
                plate_text.push(ch);
                if let Some(ref mut probs) = char_probs {
                    probs.push(slot_view[max_idx]);
                }
            }
        }

        // Debug: show raw plate text before removing padding
        println!("Raw plate text before padding removal: '{}'", plate_text);

        // Remove trailing padding characters if requested
        if remove_pad_char {
            if let Some(pad) = pad_char {
                plate_text = plate_text.trim_end_matches(pad).to_string();
            }
        }

        predictions.push(PlatePrediction {
            plate: plate_text,
            char_probs,
            region: None,
            region_prob: None,
        });
    }

    Ok(predictions)
}

/// Decode region output from model inference.
///
/// # Arguments
///
/// * `region_output` - Region logits/probabilities with shape (N, num_regions)
/// * `regions` - List of region labels
///
/// # Returns
///
/// A vector of (region_name, probability) tuples, one per input sample.
pub fn decode_region_output(
    region_output: &ArrayViewD<f32>,
    regions: &[String],
) -> Result<Vec<(String, f32)>> {
    if region_output.ndim() != 2 {
        return Err(crate::error::OcrError::InvalidShape(format!(
            "Region output must be 2D, got {} dimensions",
            region_output.ndim()
        )));
    }

    let batch_size = region_output.len_of(ndarray::Axis(0));
    let num_regions = region_output.len_of(ndarray::Axis(1));

    if num_regions != regions.len() {
        return Err(crate::error::OcrError::InvalidShape(format!(
            "Region output size {} doesn't match number of region labels {}",
            num_regions,
            regions.len()
        )));
    }

    let mut results = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let row = region_output.slice(ndarray::s![i, ..]);
        let max_idx = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let region = regions
            .get(max_idx)
            .map(|s| s.as_str())
            .unwrap_or("Unknown")
            .to_string();
        let prob = row[max_idx];

        results.push((region, prob));
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_decode_plate_output_basic() {
        // Create a mock output: batch of 2, 3 slots, vocab of 4 (ABC_)
        let mut data = Array3::zeros((2, 3, 4));
        // First sample: A B _
        data[[0, 0, 0]] = 1.0; // A
        data[[0, 1, 1]] = 1.0; // B
        data[[0, 2, 3]] = 1.0; // _
        // Second sample: C C C
        data[[1, 0, 2]] = 1.0; // C
        data[[1, 1, 2]] = 1.0; // C
        data[[1, 2, 2]] = 1.0; // C

        let output = data.into_dyn();
        let results = decode_plate_output(&output, 3, "ABC_", Some('_'), true, false).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].plate, "AB");
        assert_eq!(results[1].plate, "CCC");
    }

    #[test]
    fn test_decode_plate_output_with_confidence() {
        let mut data = Array3::zeros((1, 3, 4));
        // Sample: A B C with varying confidence
        data[[0, 0, 0]] = 0.9; // A - 90% confidence
        data[[0, 1, 1]] = 0.8; // B - 80% confidence
        data[[0, 2, 2]] = 0.7; // C - 70% confidence

        let output = data.into_dyn();
        let results =
            decode_plate_output(&output, 3, "ABC_", None, false, true).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].plate, "ABC");
        assert!(results[0].char_probs.is_some());
        let probs = results[0].char_probs.as_ref().unwrap();
        assert!((probs[0] - 0.9).abs() < 0.01);
        assert!((probs[1] - 0.8).abs() < 0.01);
        assert!((probs[2] - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_decode_region_output() {
        let mut data = Array3::zeros((2, 3));
        // First sample: highest prob at index 1
        data[[0, 0]] = 0.1;
        data[[0, 1]] = 0.8;
        data[[0, 2]] = 0.1;
        // Second sample: highest prob at index 2
        data[[1, 0]] = 0.1;
        data[[1, 1]] = 0.2;
        data[[1, 2]] = 0.7;

        let output = data.into_dyn();
        let regions = vec
!["USA".to_string(), "Europe".to_string(), "Asia".to_string()];
        let results = decode_region_output(&output, &regions).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "Europe");
        assert!((results[0].1 - 0.8).abs() < 0.01);
        assert_eq!(results[1].0, "Asia");
        assert!((results[1].1 - 0.7).abs() < 0.01);
    }
}
