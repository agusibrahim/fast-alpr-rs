//! Configuration structures for plate recognition.

use crate::error::{OcrError, Result};
use crate::types::{ColorMode, InterpolationMethod, PaddingColor};
use serde::Deserialize;
use std::path::Path;

/// Configuration for plate recognition models.
///
/// This mirrors the Python `PlateConfig` dataclass.
#[derive(Debug, Clone, Deserialize)]
pub struct PlateConfig {
    /// Maximum number of plate slots (character positions).
    pub max_plate_slots: usize,

    /// All possible characters in the model's alphabet.
    #[serde(deserialize_with = "deserialize_alphabet")]
    pub alphabet: String,

    /// Padding character for shorter plates.
    #[serde(deserialize_with = "deserialize_pad_char")]
    pub pad_char: char,

    /// Image height for model input.
    pub img_height: u32,

    /// Image width for model input.
    pub img_width: u32,

    /// Keep aspect ratio when resizing.
    #[serde(default)]
    pub keep_aspect_ratio: bool,

    /// Interpolation method for resizing.
    #[serde(default)]
    pub interpolation: InterpolationMethod,

    /// Color mode for input images.
    #[serde(default)]
    pub image_color_mode: ColorMode,

    /// Padding color for letterboxing.
    #[serde(default = "default_padding_color")]
    pub padding_color: PaddingColor,

    /// Optional region labels for region recognition.
    #[serde(default)]
    pub plate_regions: Option<Vec<String>>,
}

impl PlateConfig {
    /// Load configuration from a YAML file.
    pub fn from_yaml<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|_| OcrError::FileNotFound(path.as_ref().to_path_buf()))?;

        let config: PlateConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Get the vocabulary size (length of alphabet).
    pub fn vocabulary_size(&self) -> usize {
        self.alphabet.chars().count()
    }

    /// Get the index of the padding character in the alphabet.
    pub fn pad_idx(&self) -> usize {
        self.alphabet
            .chars()
            .position(|c| c == self.pad_char)
            .expect("pad_char must be in alphabet")
    }

    /// Get the number of channels (1 for grayscale, 3 for RGB).
    pub fn num_channels(&self) -> usize {
        match self.image_color_mode {
            ColorMode::Grayscale => 1,
            ColorMode::Rgb => 3,
        }
    }

    /// Check if this config supports region recognition.
    pub fn has_region_recognition(&self) -> bool {
        self.plate_regions
            .as_ref()
            .map(|r| !r.is_empty())
            .unwrap_or(false)
    }
}

/// Deserializer for alphabet that handles both quoted and unquoted strings.
fn deserialize_alphabet<'de, D>(deserializer: D) -> std::result::Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    Ok(s)
}

/// Deserializer for pad_char that takes the first character of the string.
fn deserialize_pad_char<'de, D>(deserializer: D) -> std::result::Result<char, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    s.chars()
        .next()
        .ok_or_else(|| serde::de::Error::custom("pad_char cannot be empty"))
}

/// Default padding color (114, 114, 114) - the gray used in YOLO.
fn default_padding_color() -> PaddingColor {
    PaddingColor::Gray(114)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_basic() {
        let yaml = r#"
max_plate_slots: 9
alphabet: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
pad_char: "_"
img_height: 64
img_width: 128
"#;
        let config: PlateConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.max_plate_slots, 9);
        assert_eq!(config.alphabet.len(), 37);
        assert_eq!(config.pad_char, '_');
        assert_eq!(config.img_height, 64);
        assert_eq!(config.img_width, 128);
    }

    #[test]
    fn test_vocabulary_size() {
        let yaml = r#"
max_plate_slots: 9
alphabet: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
pad_char: "_"
img_height: 64
img_width: 128
"#;
        let config: PlateConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.vocabulary_size(), 37);
    }

    #[test]
    fn test_pad_idx() {
        let yaml = r#"
max_plate_slots: 9
alphabet: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
pad_char: "_"
img_height: 64
img_width: 128
"#;
        let config: PlateConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.pad_idx(), 36);
    }
}
