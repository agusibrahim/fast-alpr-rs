use axum::{
    extract::{Multipart, State, DefaultBodyLimit},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use fast_plate_ocr::{ALPR, PlateConfig};
use serde::Serialize;
use std::path::Path;
use std::sync::{Arc, Mutex};
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;
use tokio::net::TcpListener;
use tokio::task::spawn_blocking;

const DEFAULT_DETECTOR: &str = "models/yolo-v9-t-416-license-plates-end2end.onnx";
const DEFAULT_OCR_MODEL: &str = "models/cct_xs_v2_global.onnx";
const DEFAULT_OCR_CONFIG: &str = "models/cct_xs_v2_global_plate_config.yaml";

/// Application state holding our default ALPR instance
struct AppState {
    alpr: Mutex<ALPR>,
}

#[derive(Serialize, ToSchema)]
pub struct ApiResponse {
    pub success: bool,
    pub plates: Vec<PlateResult>,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    pub error: Option<String>,
}

#[derive(Serialize, ToSchema)]
pub struct PlateResult {
    pub text: String,
    pub confidence: f32,
    pub region: Option<String>,
    pub bbox: BoundingBox,
}

#[derive(Serialize, ToSchema)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Serialize, ToSchema)]
pub struct ModelInfo {
    pub name: String,
    pub path: String,
    pub size_bytes: u64,
    pub model_type: String,
}

#[derive(Serialize, ToSchema)]
pub struct ModelsResponse {
    pub detection: Vec<ModelInfo>,
    pub ocr: Vec<ModelInfo>,
}

#[derive(OpenApi)]
#[openapi(
    paths(alpr_detect, list_models),
    components(schemas(ApiResponse, PlateResult, BoundingBox, UploadImage, ModelsResponse, ModelInfo)),
    tags((name = "ALPR", description = "Automatic License Plate Recognition API"))
)]
struct ApiDoc;

pub async fn serve(port: u16) -> anyhow::Result<()> {
    println!("Loading config from: {}", DEFAULT_OCR_CONFIG);
    let config = PlateConfig::from_yaml(DEFAULT_OCR_CONFIG)?;

    println!("Loading ALPR models...");
    let alpr = ALPR::new(DEFAULT_DETECTOR, 0.4, DEFAULT_OCR_MODEL, &config)?;
    let app_state = Arc::new(AppState {
        alpr: Mutex::new(alpr),
    });

    let app = Router::new()
        .route("/api/v1/alpr", post(alpr_detect))
        .route("/api/v1/models", get(list_models))
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .layer(DefaultBodyLimit::max(20 * 1024 * 1024))
        .with_state(app_state);

    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr).await?;
    println!("Server running on http://{}", addr);
    println!("Swagger UI available at http://{}/swagger-ui", addr);

    axum::serve(listener, app).await?;
    Ok(())
}

/// List available models in the models/ directory.
#[utoipa::path(
    get,
    path = "/api/v1/models",
    responses(
        (status = 200, description = "List of available models", body = ModelsResponse)
    ),
    tag = "ALPR"
)]
async fn list_models() -> impl IntoResponse {
    let mut detection = Vec::new();
    let mut ocr = Vec::new();

    if let Ok(entries) = std::fs::read_dir("models") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("onnx") {
                continue;
            }
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            let size_bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
            let rel_path = format!("models/{}", name);

            let info = ModelInfo {
                name: name.clone(),
                path: rel_path,
                size_bytes,
                model_type: if name.contains("yolo") { "detection".into() } else { "ocr".into() },
            };

            if name.contains("yolo") {
                detection.push(info);
            } else {
                ocr.push(info);
            }
        }
    }

    detection.sort_by(|a, b| a.name.cmp(&b.name));
    ocr.sort_by(|a, b| a.name.cmp(&b.name));

    Json(ModelsResponse { detection, ocr })
}

/// Auto-detect OCR config YAML path from OCR model path.
fn find_ocr_config(ocr_model_path: &str) -> Option<String> {
    let p = Path::new(ocr_model_path);
    let stem = p.file_stem()?.to_string_lossy();
    let dir = p.parent().unwrap_or(Path::new("."));

    // Try _plate_config.yaml first, then _config.yaml
    for suffix in &["_plate_config.yaml", "_config.yaml"] {
        let candidate = dir.join(format!("{}{}", stem, suffix));
        if candidate.exists() {
            return Some(candidate.to_string_lossy().to_string());
        }
    }
    None
}

/// Detect license plates from an uploaded image.
///
/// Optionally specify custom models via `detector_model` and `ocr_model` fields
/// (e.g. `models/yolo-v9-t-512-license-plates-end2end.onnx`).
/// Use GET `/api/v1/models` to list available models.
#[utoipa::path(
    post,
    path = "/api/v1/alpr",
    request_body(content = UploadImage, description = "Image file and optional model paths", content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "Successful detection", body = ApiResponse),
        (status = 400, description = "Bad request", body = ApiResponse)
    ),
    tag = "ALPR"
)]
async fn alpr_detect(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let mut image_bytes = None;
    let mut detector_model: Option<String> = None;
    let mut ocr_model: Option<String> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        match field.name() {
            Some("image") => {
                if let Ok(bytes) = field.bytes().await {
                    image_bytes = Some(bytes);
                }
            }
            Some("detector_model") => {
                if let Ok(text) = field.text().await {
                    if !text.is_empty() {
                        detector_model = Some(text);
                    }
                }
            }
            Some("ocr_model") => {
                if let Ok(text) = field.text().await {
                    if !text.is_empty() {
                        ocr_model = Some(text);
                    }
                }
            }
            _ => {}
        }
    }

    let bytes = match image_bytes {
        Some(b) => b,
        None => return (StatusCode::BAD_REQUEST, Json(ApiResponse {
            success: false,
            plates: vec![],
            processing_time_ms: 0.0,
            error: Some("Missing 'image' field in multipart".to_string()),
        })),
    };

    let use_custom = detector_model.is_some() || ocr_model.is_some();
    let start = std::time::Instant::now();

    let result = spawn_blocking(move || {
        // Save bytes to temp file
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
            
        // Guess the format from the bytes to determine the correct extension
        let ext = match image::guess_format(&bytes) {
            Ok(format) => format.extensions_str().first().unwrap_or(&"jpg").to_string(),
            Err(_) => "jpg".to_string(), // Fallback
        };
        
        let temp_file = std::env::temp_dir().join(format!("alpr_upload_{}.{}", nanos, ext));
        if let Err(e) = std::fs::write(&temp_file, &bytes) {
            return Err(e.to_string());
        }

        let prediction = if use_custom {
            // Create ALPR with custom models
            let det = detector_model.as_deref().unwrap_or(DEFAULT_DETECTOR);
            let ocr = ocr_model.as_deref().unwrap_or(DEFAULT_OCR_MODEL);

            let config_path = find_ocr_config(ocr)
                .unwrap_or_else(|| DEFAULT_OCR_CONFIG.to_string());

            let config = PlateConfig::from_yaml(&config_path)
                .map_err(|e| format!("Config error for '{}': {}", config_path, e))?;

            let mut alpr = ALPR::new(det, 0.4, ocr, &config)
                .map_err(|e| format!("Model load error: {}", e))?;

            alpr.predict(&temp_file).map_err(|e| e.to_string())
        } else {
            // Use default ALPR
            let mut alpr = state.alpr.lock().unwrap();
            alpr.predict(&temp_file).map_err(|e| e.to_string())
        };

        let _ = std::fs::remove_file(temp_file);
        prediction
    }).await.unwrap_or_else(|e| Err(e.to_string()));

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    match result {
        Ok(results) => {
            let mut plates: Vec<PlateResult> = results.into_iter().map(|r| {
                let (text, region) = if let Some(ocr) = r.ocr {
                    (ocr.plate, ocr.region)
                } else {
                    ("".to_string(), None)
                };

                PlateResult {
                    text,
                    confidence: r.detection.confidence,
                    region,
                    bbox: BoundingBox {
                        x1: r.detection.bounding_box.x1 as f32,
                        y1: r.detection.bounding_box.y1 as f32,
                        x2: r.detection.bounding_box.x2 as f32,
                        y2: r.detection.bounding_box.y2 as f32,
                    }
                }
            }).collect();

            // Sort plates by combined score: (Bounding Box Area) * Confidence in descending order
            plates.sort_by(|a, b| {
                let area_a = (a.bbox.x2 - a.bbox.x1) * (a.bbox.y2 - a.bbox.y1);
                let score_a = area_a * a.confidence;
                
                let area_b = (b.bbox.x2 - b.bbox.x1) * (b.bbox.y2 - b.bbox.y1);
                let score_b = area_b * b.confidence;
                
                score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
            });

            (StatusCode::OK, Json(ApiResponse {
                success: true,
                plates,
                processing_time_ms: elapsed_ms,
                error: None,
            }))
        },
        Err(e) => {
            (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse {
                success: false,
                plates: vec![],
                processing_time_ms: elapsed_ms,
                error: Some(e),
            }))
        }
    }
}

/// Dummy schema for utoipa to generate multipart properly
#[allow(dead_code)]
#[derive(ToSchema)]
struct UploadImage {
    /// Image file (JPEG/PNG/WebP)
    #[schema(value_type = String, format = Binary)]
    image: Vec<u8>,
    /// Optional detection model path (e.g. "models/yolo-v9-t-512-license-plates-end2end.onnx")
    #[schema(value_type = Option<String>, example = "models/yolo-v9-t-416-license-plates-end2end.onnx")]
    detector_model: Option<String>,
    /// Optional OCR model path (e.g. "models/cct_xs_v2_global.onnx"). Config YAML is auto-detected.
    #[schema(value_type = Option<String>, example = "models/cct_xs_v2_global.onnx")]
    ocr_model: Option<String>,
}
