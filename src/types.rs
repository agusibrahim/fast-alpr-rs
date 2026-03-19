//! Common type definitions for the library.

use serde::Deserialize;
use std::fmt;

/// Interpolation method for image resizing.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum InterpolationMethod {
    /// Nearest neighbor interpolation.
    Nearest,
    /// Linear interpolation (default).
    #[default]
    Linear,
    /// Cubic interpolation.
    Cubic,
    /// Area interpolation.
    Area,
    /// Lanczos4 interpolation.
    Lanczos4,
}

impl InterpolationMethod {
    /// Convert to image crate's FilterType.
    pub fn to_filter_type(&self) -> image::imageops::FilterType {
        match self {
            Self::Nearest => image::imageops::FilterType::Nearest,
            Self::Linear => image::imageops::FilterType::Triangle,
            Self::Cubic => image::imageops::FilterType::CatmullRom,
            Self::Area => image::imageops::FilterType::Gaussian,
            Self::Lanczos4 => image::imageops::FilterType::Lanczos3,
        }
    }

    /// Parse from string (for compatibility with YAML).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "nearest" => Some(Self::Nearest),
            "linear" => Some(Self::Linear),
            "cubic" => Some(Self::Cubic),
            "area" => Some(Self::Area),
            "lanczos4" => Some(Self::Lanczos4),
            _ => None,
        }
    }
}

impl fmt::Display for InterpolationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nearest => write!(f, "nearest"),
            Self::Linear => write!(f, "linear"),
            Self::Cubic => write!(f, "cubic"),
            Self::Area => write!(f, "area"),
            Self::Lanczos4 => write!(f, "lanczos4"),
        }
    }
}

/// Color mode for input images.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ColorMode {
    /// Grayscale (single channel).
    Grayscale,
    /// RGB (three channels).
    #[default]
    Rgb,
}

impl ColorMode {
    /// Get the number of channels for this color mode.
    pub fn num_channels(&self) -> usize {
        match self {
            Self::Grayscale => 1,
            Self::Rgb => 3,
        }
    }
}

impl fmt::Display for ColorMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Grayscale => write!(f, "grayscale"),
            Self::Rgb => write!(f, "rgb"),
        }
    }
}

/// Padding color for letterboxing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingColor {
    /// Single value for grayscale.
    Gray(u8),
    /// RGB triplet.
    Rgb(u8, u8, u8),
}

impl PaddingColor {
    /// Get as grayscale value.
    pub fn as_gray(&self) -> u8 {
        match self {
            Self::Gray(v) => *v,
            Self::Rgb(r, g, b) => ((u16::from(*r) + u16::from(*g) + u16::from(*b)) / 3) as u8,
        }
    }

    /// Get as RGB tuple.
    pub fn as_rgb(&self) -> (u8, u8, u8) {
        match self {
            Self::Gray(v) => (*v, *v, *v),
            Self::Rgb(r, g, b) => (*r, *g, *b),
        }
    }
}

impl<'de> Deserialize<'de> for PaddingColor {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct PaddingColorVisitor;

        impl<'de> serde::de::Visitor<'de> for PaddingColorVisitor {
            type Value = PaddingColor;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("an integer or an array of 3 integers")
            }

            fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(PaddingColor::Gray(value as u8))
            }

            fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(PaddingColor::Gray(value as u8))
            }

            fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut values = Vec::with_capacity(3);
                while let Some(value) = seq.next_element()? {
                    values.push(value);
                }
                if values.len() != 3 {
                    return Err(serde::de::Error::custom(
                        "padding_color array must have exactly 3 elements",
                    ));
                }
                Ok(PaddingColor::Rgb(values[0], values[1], values[2]))
            }
        }

        deserializer.deserialize_any(PaddingColorVisitor)
    }
}

/// Result of plate recognition inference.
#[derive(Debug, Clone)]
pub struct PlatePrediction {
    /// Decoded license plate text.
    pub plate: String,

    /// Optional per-character confidence scores.
    pub char_probs: Option<Vec<f32>>,

    /// Optional predicted region label.
    pub region: Option<String>,

    /// Optional probability for the predicted region.
    pub region_prob: Option<f32>,
}

impl PlatePrediction {
    /// Create a new prediction with just the plate text.
    pub fn new(plate: impl Into<String>) -> Self {
        Self {
            plate: plate.into(),
            char_probs: None,
            region: None,
            region_prob: None,
        }
    }

    /// Check if this prediction has confidence scores.
    pub fn has_confidence(&self) -> bool {
        self.char_probs.is_some()
    }

    /// Check if this prediction has region information.
    pub fn has_region(&self) -> bool {
        self.region.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolation_method() {
        assert_eq!(InterpolationMethod::from_str("nearest"), Some(InterpolationMethod::Nearest));
        assert_eq!(InterpolationMethod::from_str("linear"), Some(InterpolationMethod::Linear));
        assert_eq!(InterpolationMethod::from_str("cubic"), Some(InterpolationMethod::Cubic));
        assert_eq!(InterpolationMethod::from_str("area"), Some(InterpolationMethod::Area));
        assert_eq!(InterpolationMethod::from_str("lanczos4"), Some(InterpolationMethod::Lanczos4));
        assert_eq!(InterpolationMethod::from_str("invalid"), None);
    }

    #[test]
    fn test_padding_color() {
        let yaml_single = "114";
        let color: PaddingColor = serde_yaml::from_str(yaml_single).unwrap();
        assert_eq!(color, PaddingColor::Gray(114));

        let yaml_array = "[114, 114, 114]";
        let color: PaddingColor = serde_yaml::from_str(yaml_array).unwrap();
        assert_eq!(color, PaddingColor::Rgb(114, 114, 114));
    }
}
