//! Error types for fast-plate-ocr-rust.

use std::path::PathBuf;

use image::ImageError;
use ndarray::ShapeError;
use thiserror::Error;

/// Main error type for the library.
#[derive(Debug, Error)]
pub enum OcrError {
    /// Error loading or processing an image.
    #[error("Image load error: {0}")]
    ImageLoad(#[from] ImageError),

    /// Error from ONNX Runtime.
    #[error("ONNX Runtime error: {0}")]
    Onnx(#[from] ort::Error),

    /// Error related to configuration.
    #[error("Config error: {0}")]
    Config(String),

    /// Invalid output shape from model.
    #[error("Invalid output shape: {0}")]
    InvalidShape(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// File not found.
    #[error("File not found: {0}")]
    FileNotFound(PathBuf),

    /// YAML parsing error.
    #[error("YAML parsing error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    /// Invalid interpolation method.
    #[error("Invalid interpolation method: {0}")]
    InvalidInterpolation(String),

    /// Invalid color mode.
    #[error("Invalid color mode: {0}")]
    InvalidColorMode(String),

    /// Invalid padding color.
    #[error("Invalid padding color: {0}")]
    InvalidPaddingColor(String),

    /// Array shape error.
    #[error("Array shape error: {0}")]
    ArrayShape(#[from] ShapeError),
}

/// Result type alias for library operations.
pub type Result<T> = std::result::Result<T, OcrError>;
