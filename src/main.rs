//! Fast Plate OCR & ALPR - Rust Implementation

use fast_plate_ocr::{ALPR, LicensePlateRecognizer, PlateConfig};
use std::path::PathBuf;

mod api;

#[tokio::main]
async fn main() -> fast_plate_ocr::Result<()> {
    tracing_subscriber::fmt::init();
    
    println!("Fast Plate OCR & ALPR - Rust Implementation");
    println!("===========================================\n");

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let mode = &args[1];

    match mode.as_str() {
        "ocr" => run_ocr(&args[1..]),
        "alpr" => run_alpr(&args[1..]),
        "serve" => {
            if let Err(e) = api::serve().await {
                eprintln!("Server error: {}", e);
                std::process::exit(1);
            }
            Ok(())
        }
        _ => {
            println!("Unknown mode: {}", mode);
            print_usage();
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    println!("Usage: fast_plate_ocr <mode> [options] <image1> [image2] ...\n");
    println!("Modes:");
    println!("  serve");
    println!("        Start the REST API server on port 3000");
    println!();
    println!("  ocr  [model.onnx] [config.yaml] <image> [images...]");
    println!("        Run OCR on pre-cropped license plate images.");
    println!("        If models are omitted, default local models will be used.");
    println!();
    println!("  alpr [detector.onnx] [ocr.onnx] [config.yaml] <image> [images...]");
    println!("        Run full ALPR (detection + OCR) on traffic scene images.");
    println!("        If models are omitted, default local models will be used.");
    println!();
    println!("Examples:");
    println!("  # Start server");
    println!("  fast_plate_ocr serve");
    println!();
    println!("  # OCR with default models");
    println!("  fast_plate_ocr ocr plate.webp");
    println!();
    println!("  # Full ALPR with default models");
    println!("  fast_plate_ocr alpr scene.jpg");
}

fn run_ocr(args: &[String]) -> fast_plate_ocr::Result<()> {
    // If user passed <model.onnx> <config.yaml> <img1> ..., len >= 4
    // If user just passed <img1> ..., len >= 2
    let (model_path, config_path, image_paths) = if args.len() >= 4 && args[1].ends_with(".onnx") && args[2].ends_with(".yaml") {
        (
            args[1].clone(),
            args[2].clone(),
            args[3..].iter().map(|s| PathBuf::from(s)).collect::<Vec<_>>()
        )
    } else if args.len() >= 2 {
        println!("Using default OCR models.");
        (
            "models/cct_xs_v2_global.onnx".to_string(),
            "models/cct_xs_v2_global_plate_config.yaml".to_string(),
            args[1..].iter().map(|s| PathBuf::from(s)).collect::<Vec<_>>()
        )
    } else {
        println!("Usage: fast_plate_ocr ocr [model.onnx] [config.yaml] <images...>");
        std::process::exit(1);
    };

    // Load configuration
    println!("Loading config from: {}", config_path);
    let config = PlateConfig::from_yaml(&config_path)?;
    println!("  - Max plate slots: {}", config.max_plate_slots);
    println!("  - Alphabet size: {}", config.vocabulary_size());
    println!("  - Image size: {}x{}", config.img_width, config.img_height);
    println!("  - Color mode: {}", config.image_color_mode);
    println!("  - Region recognition: {}", config.has_region_recognition());

    // Create recognizer
    println!("\nLoading OCR model from: {}", model_path);
    let mut recognizer = LicensePlateRecognizer::new(model_path, &config)?;
    println!("  - Output names: {:?}", recognizer.output_names());

    // Run inference
    if image_paths.is_empty() {
        println!("\nNo images provided. Exiting.");
        return Ok(());
    }

    println!("\nProcessing {} image(s)...", image_paths.len());

    for path in &image_paths {
        println!("\nImage: {}", path.display());
        match recognizer.run(path) {
            Ok(result) => {
                println!("  Plate: {}", result.plate);
                if result.has_confidence() {
                    println!("  Confidence scores available");
                }
                if result.has_region() {
                    println!("  Region: {}", result.region.as_ref().unwrap());
                }
            }
            Err(e) => {
                eprintln!("  Error: {}", e);
            }
        }
    }

    Ok(())
}

fn run_alpr(args: &[String]) -> fast_plate_ocr::Result<()> {
    // If user passed <detector.onnx> <ocr.onnx> <config.yaml> <img1> ..., len >= 5
    // If user just passed <img1> ..., len >= 2
    let (detector_path, ocr_model_path, config_path, image_paths) = if args.len() >= 5 && args[1].ends_with(".onnx") && args[2].ends_with(".onnx") && args[3].ends_with(".yaml") {
        (
            args[1].clone(),
            args[2].clone(),
            args[3].clone(),
            args[4..].iter().map(|s| PathBuf::from(s)).collect::<Vec<_>>()
        )
    } else if args.len() >= 2 {
        println!("Using default ALPR models.");
        (
            "models/yolo-v9-t-416-license-plates-end2end.onnx".to_string(),
            "models/cct_xs_v2_global.onnx".to_string(),
            "models/cct_xs_v2_global_plate_config.yaml".to_string(),
            args[1..].iter().map(|s| PathBuf::from(s)).collect::<Vec<_>>()
        )
    } else {
        println!("Usage: fast_plate_ocr alpr [detector.onnx] [ocr.onnx] [config.yaml] <images...>");
        std::process::exit(1);
    };

    // Load configuration
    println!("Loading config from: {}", config_path);
    let config = PlateConfig::from_yaml(&config_path)?;
    println!("  - Max plate slots: {}", config.max_plate_slots);
    println!("  - Alphabet size: {}", config.vocabulary_size());
    println!("  - Image size: {}x{}", config.img_width, config.img_height);
    println!("  - Color mode: {}", config.image_color_mode);

    // Create ALPR system
    println!("\nLoading detector from: {}", detector_path);
    println!("Loading OCR model from: {}", ocr_model_path);
    let mut alpr = ALPR::new(
        detector_path,
        0.4,  // confidence threshold
        ocr_model_path,
        &config,
    )?;

    // Run inference
    if image_paths.is_empty() {
        println!("\nNo images provided. Exiting.");
        return Ok(());
    }

    println!("\nProcessing {} image(s)...", image_paths.len());

    for path in &image_paths {
        println!("\n========================================");
        println!("Image: {}", path.display());
        println!("========================================");

        match alpr.predict(path) {
            Ok(results) => {
                if results.is_empty() {
                    println!("No license plates detected.");
                } else {
                    println!("Found {} license plate(s):", results.len());
                    for (i, result) in results.iter().enumerate() {
                        println!();
                        println!("Plate #{}:", i + 1);
                        println!("  Detection confidence: {:.2}%", result.detection.confidence * 100.0);
                        println!("  Bounding box: ({}, {}) -> ({}, {})",
                            result.detection.bounding_box.x1,
                            result.detection.bounding_box.y1,
                            result.detection.bounding_box.x2,
                            result.detection.bounding_box.y2,
                        );

                        if let Some(ref ocr) = result.ocr {
                            println!("  Plate text: {}", ocr.plate);
                            if let Some(ref region) = ocr.region {
                                println!("  Region: {}", region);
                            }
                        } else {
                            println!("  Plate text: (OCR failed)");
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
    }

    Ok(())
}
