//! Image preprocessing for plate recognition.

use crate::config::PlateConfig;
use crate::error::Result;
use crate::types::{ColorMode, PaddingColor};
use image::{DynamicImage, GrayImage, Luma, Rgb, RgbImage};
use ndarray::Array4;
use std::path::Path;

/// Load and resize an image from a file path.
///
/// This function reads an image from disk and resizes it according to the configuration.
pub fn load_and_resize_image<P: AsRef<Path>>(
    path: P,
    config: &PlateConfig,
) -> Result<DynamicImage> {
    let img = image::open(path.as_ref())?;
    resize_image(&img, config)
}

/// Resize an image according to the configuration.
///
/// This handles aspect ratio preservation, color mode conversion, and letterboxing.
pub fn resize_image(img: &DynamicImage, config: &PlateConfig) -> Result<DynamicImage> {
    let (target_w, target_h) = (config.img_width, config.img_height);

    println!("Original image size: {}x{}", img.width(), img.height());
    println!("Target size: {}x{}", target_w, target_h);

    if !config.keep_aspect_ratio {
        // Simple resize without aspect ratio preservation
        println!("Resizing without aspect ratio preservation...");
        // Use resize_exact to force exact dimensions without aspect ratio preservation
        let resized = img.resize_exact(target_w, target_h, config.interpolation.to_filter_type());
        println!("Resized size: {}x{}", resized.width(), resized.height());
        Ok(convert_color_mode(resized, config.image_color_mode))
    } else {
        // Resize with aspect ratio preservation (letterboxing)
        let orig_w = img.width();
        let orig_h = img.height();

        // Calculate scale ratio
        let r = (target_w as f32 / orig_w as f32).min(target_h as f32 / orig_h as f32);
        let new_unpad_w = (orig_w as f32 * r).round() as u32;
        let new_unpad_h = (orig_h as f32 * r).round() as u32;

        // Resize if necessary
        let resized = if (new_unpad_w, new_unpad_h) != (orig_w, orig_h) {
            img.resize(new_unpad_w, new_unpad_h, config.interpolation.to_filter_type())
        } else {
            img.clone()
        };

        // Calculate padding
        let dw = (target_w as f32 - new_unpad_w as f32) / 2.0;
        let dh = (target_h as f32 - new_unpad_h as f32) / 2.0;

        let top = dh.floor() as u32;
        let bottom = dh.ceil() as u32;
        let left = dw.floor() as u32;
        let right = dw.ceil() as u32;

        // Apply letterbox padding
        let letterboxed = apply_letterbox(
            &resized,
            target_w,
            target_h,
            top,
            bottom,
            left,
            right,
            config.padding_color,
            config.image_color_mode,
        );

        Ok(convert_color_mode(letterboxed, config.image_color_mode))
    }
}

/// Convert an image to the specified color mode.
fn convert_color_mode(img: DynamicImage, color_mode: ColorMode) -> DynamicImage {
    match color_mode {
        ColorMode::Grayscale => img.to_luma8().into(),
        ColorMode::Rgb => img.to_rgb8().into(),
    }
}

/// Apply letterbox padding to an image.
fn apply_letterbox(
    img: &DynamicImage,
    target_w: u32,
    target_h: u32,
    top: u32,
    bottom: u32,
    left: u32,
    right: u32,
    padding_color: PaddingColor,
    color_mode: ColorMode,
) -> DynamicImage {
    match color_mode {
        ColorMode::Grayscale => {
            let gray_img = img.to_luma8();
            let padded_w = gray_img.width() + left + right;
            let padded_h = gray_img.height() + top + bottom;

            let mut padded = GrayImage::new(padded_w, padded_h);
            let fill = Luma([padding_color.as_gray()]);

            // Fill with padding color
            for y in 0..padded_h {
                for x in 0..padded_w {
                    padded.put_pixel(x, y, fill);
                }
            }

            // Copy original image
            for y in 0..gray_img.height() {
                for x in 0..gray_img.width() {
                    let pixel = gray_img.get_pixel(x, y);
                    padded.put_pixel(x + left, y + top, *pixel);
                }
            }

            // Crop to exact target size if needed
            if padded_w != target_w || padded_h != target_h {
                padded = image::imageops::crop(&mut padded, 0, 0, target_w, target_h).to_image();
            }

            padded.into()
        }
        ColorMode::Rgb => {
            let rgb_img = img.to_rgb8();
            let padded_w = rgb_img.width() + left + right;
            let padded_h = rgb_img.height() + top + bottom;

            let mut padded = RgbImage::new(padded_w, padded_h);
            let (pr, pg, pb) = padding_color.as_rgb();
            let fill = Rgb([pr, pg, pb]);

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

            // Crop to exact target size if needed
            if padded_w != target_w || padded_h != target_h {
                padded = image::imageops::crop(&mut padded, 0, 0, target_w, target_h).to_image();
            }

            padded.into()
        }
    }
}

/// Convert an image to a tensor for model input.
///
/// Returns a tensor with shape (1, H, W, C) in channels-last format.
pub fn image_to_tensor(img: &DynamicImage, config: &PlateConfig) -> Result<Array4<u8>> {
    let (h, w) = (config.img_height as usize, config.img_width as usize);
    let c = config.num_channels();

    println!("Converting image to tensor: H={}, W={}, C={}", h, w, c);

    let pixels = match config.image_color_mode {
        ColorMode::Grayscale => {
            println!("Converting to grayscale...");
            let gray_img = img.to_luma8();
            let collected: Vec<u8> = gray_img.pixels().map(|p| p[0]).collect();
            println!("Grayscale pixels: {}", collected.len());
            collected
        }
        ColorMode::Rgb => {
            println!("Converting to RGB...");
            let rgb_img = img.to_rgb8();
            let collected: Vec<u8> = rgb_img.pixels().flat_map(|p| vec![p[0], p[1], p[2]]).collect();
            println!("RGB pixels: {}", collected.len());
            collected
        }
    };

    let expected_size = h * w * c;
    println!("Expected pixel count: {}", expected_size);
    println!("Actual pixel count: {}", pixels.len());

    if pixels.len() != expected_size {
        return Err(crate::error::OcrError::InvalidShape(format!(
            "Pixel count mismatch: expected {}, got {}",
            expected_size, pixels.len()
        )));
    }

    // Create array with shape (1, H, W, C)
    let tensor = Array4::from_shape_vec((1, h, w, c), pixels)?;

    Ok(tensor)
}

/// Preprocess a batch of images for model input.
///
/// Ensures the output is always a 4D array with shape (N, H, W, C).
pub fn preprocess_image(img: &DynamicImage, config: &PlateConfig) -> Result<Array4<u8>> {
    let resized = resize_image(img, config)?;
    image_to_tensor(&resized, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resize_simple() {
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

        // Create a simple test image
        let img = DynamicImage::new_rgb8(256, 128);
        let resized = resize_image(&img, &config).unwrap();

        assert_eq!(resized.width(), 128);
        assert_eq!(resized.height(), 64);
    }

    #[test]
    fn test_resize_aspect_ratio() {
        let config = PlateConfig {
            max_plate_slots: 9,
            alphabet: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_".to_string(),
            pad_char: '_',
            img_height: 64,
            img_width: 128,
            keep_aspect_ratio: true,
            interpolation: InterpolationMethod::Linear,
            image_color_mode: ColorMode::Rgb,
            padding_color: PaddingColor::Gray(114),
            plate_regions: None,
        };

        // Create a wide test image (2:1 aspect ratio)
        let img = DynamicImage::new_rgb8(256, 128);
        let resized = resize_image(&img, &config).unwrap();

        assert_eq!(resized.width(), 128);
        assert_eq!(resized.height(), 64);
    }

    #[test]
    fn test_image_to_tensor() {
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

        let img = DynamicImage::new_rgb8(128, 64);
        let tensor = image_to_tensor(&img, &config).unwrap();

        assert_eq!(tensor.shape(), &[1, 64, 128, 3]);
    }

    #[test]
    fn test_image_to_tensor_grayscale() {
        let config = PlateConfig {
            max_plate_slots: 9,
            alphabet: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_".to_string(),
            pad_char: '_',
            img_height: 64,
            img_width: 128,
            keep_aspect_ratio: false,
            interpolation: InterpolationMethod::Linear,
            image_color_mode: ColorMode::Grayscale,
            padding_color: PaddingColor::Gray(114),
            plate_regions: None,
        };

        let img = DynamicImage::new_rgb8(128, 64);
        let tensor = image_to_tensor(&img, &config).unwrap();

        assert_eq!(tensor.shape(), &[1, 64, 128, 1]);
    }
}
